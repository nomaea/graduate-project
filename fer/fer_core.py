import logging
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .fer_api import FERPacket, FERQueuePublisher

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


LABELS = ["neutral", "happy", "sad", "angry"]
IMG_SIZE = 224

# EAR 기반 졸음 감지 설정
EAR_CLOSE_THRESHOLD = 0.20   # 이 값 미만이면 눈 감은 것으로 판정
EAR_DROWSY_SECONDS  = 2.0    # 연속 눈 감음 유지 시간 임계값 (초)

# MediaPipe FaceMesh 눈 랜드마크 인덱스 (p1~p6, 시계 방향)
_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# [FER#4] 0.65 → 0.4: 새 감지 결과가 약 3프레임 이내에 50% 이상 반영됨
PROB_EMA = 0.4

# [FER#2] 이 프레임 수 이상 얼굴이 없으면 prob_ema 리셋
NO_FACE_RESET_FRAMES = 5

_logger = logging.getLogger(__name__)


def _euclidean(p1, p2) -> float:
    return float(np.linalg.norm(p1 - p2))


def compute_ear(landmarks, indices: list, frame_w: int, frame_h: int) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    pts = np.array(
        [[landmarks[i].x * frame_w, landmarks[i].y * frame_h] for i in indices],
        dtype=np.float32,
    )
    vertical_1 = _euclidean(pts[1], pts[5])
    vertical_2 = _euclidean(pts[2], pts[4])
    horizontal = _euclidean(pts[0], pts[3])
    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def rotate_image(image: np.ndarray, angle_deg: float, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, rot_mat


def transform_point(x: float, y: float, rot_mat: np.ndarray):
    px = rot_mat[0, 0] * x + rot_mat[0, 1] * y + rot_mat[0, 2]
    py = rot_mat[1, 0] * x + rot_mat[1, 1] * y + rot_mat[1, 2]
    return px, py


def make_square_box(x1, y1, x2, y2, frame_w, frame_h, scale=1.45):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale

    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(frame_w, nx2)
    ny2 = min(frame_h, ny2)
    return nx1, ny1, nx2, ny2


def _pad_to_square(roi: np.ndarray) -> np.ndarray:
    """[FER#3] 프레임 경계 클램핑으로 비정방형이 된 ROI를 제로 패딩으로 정방형으로 만든다.
    cv2.resize(non-square → 224x224) 시 얼굴 왜곡 방지.
    """
    h, w = roi.shape[:2]
    if h == w:
        return roi
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=roi.dtype)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = roi
    return padded


def preprocess_roi_bgr(roi_bgr: np.ndarray):
    roi_resized = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    x = roi_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.expand_dims(x, axis=0)  # NHWC
    return x, roi_resized


def get_mesh_aligned_roi(frame_bgr: np.ndarray, face_landmarks, scale=1.45):
    fh, fw = frame_bgr.shape[:2]

    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263

    pts = []
    for lm in face_landmarks.landmark:
        x = lm.x * fw
        y = lm.y * fh
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    lx, ly = pts[LEFT_EYE_OUTER]
    rx, ry = pts[RIGHT_EYE_OUTER]

    angle      = np.degrees(np.arctan2(ry - ly, rx - lx))
    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    rotated, rot_mat = rotate_image(frame_bgr, angle, eye_center)
    rot_pts = np.array(
        [transform_point(x, y, rot_mat) for x, y in pts], dtype=np.float32
    )

    min_x = float(np.min(rot_pts[:, 0]))
    max_x = float(np.max(rot_pts[:, 0]))
    min_y = float(np.min(rot_pts[:, 1]))
    max_y = float(np.max(rot_pts[:, 1]))

    x1, y1, x2, y2 = make_square_box(min_x, min_y, max_x, max_y, fw, fh, scale=scale)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_bgr = rotated[y1:y2, x1:x2]
    roi_bgr = _pad_to_square(roi_bgr)  # [FER#3] 정방형 패딩 적용
    return roi_bgr


class FERCore:
    def __init__(
        self,
        tflite_path: str,
        publisher: Optional[FERQueuePublisher] = None,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
    ):
        self.publisher = publisher
        self.seq = 0
        self.prob_ema = None
        self._no_face_count = 0  # [FER#2] 연속 얼굴 미감지 프레임 카운터
        self._latency_samples: list = []

        # EAR 기반 졸음 상태 추적
        self._eye_closed_since: Optional[float] = None  # 눈 감기 시작 시각 (time.time())
        self._is_drowsy: bool = False

        self.interpreter = Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

    def process_frame(self, frame_bgr: np.ndarray, return_debug: bool = False):
        frame_ts = time.time()  # [FER#6] monotonic → time.time() (BIO와 기준 통일)

        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        probs_out    = None
        face_detected = False
        roi_preview  = None
        ear_value: Optional[float] = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # EAR 계산 및 졸음 상태 갱신
            fh, fw = frame_bgr.shape[:2]
            lm = face_landmarks.landmark
            left_ear  = compute_ear(lm, _LEFT_EYE_IDX,  fw, fh)
            right_ear = compute_ear(lm, _RIGHT_EYE_IDX, fw, fh)
            ear_value = (left_ear + right_ear) / 2.0

            if ear_value < EAR_CLOSE_THRESHOLD:
                if self._eye_closed_since is None:
                    self._eye_closed_since = frame_ts
                elif (frame_ts - self._eye_closed_since) >= EAR_DROWSY_SECONDS:
                    self._is_drowsy = True
            else:
                self._eye_closed_since = None
                self._is_drowsy = False

            aligned_roi = get_mesh_aligned_roi(frame_bgr, face_landmarks, scale=1.45)

            if aligned_roi is not None and aligned_roi.size > 0:
                face_detected        = True
                self._no_face_count  = 0  # [FER#2] 얼굴 감지 시 카운터 리셋
                x_nhwc, roi_preview  = preprocess_roi_bgr(aligned_roi)
                x_in = x_nhwc.astype(self.input_details["dtype"])

                try:  # [FER#1] TFLite 추론 예외 처리 — 크래시 방지
                    self.interpreter.set_tensor(self.input_details["index"], x_in)
                    self.interpreter.invoke()
                    out   = self.interpreter.get_tensor(self.output_details["index"])[0]
                    probs = softmax(out)
                except Exception as exc:
                    _logger.warning("[FER] TFLite inference failed, skipping frame: %s", exc)
                    probs = None

                if probs is not None:
                    if self.prob_ema is None:
                        self.prob_ema = probs.copy()
                    else:
                        self.prob_ema = PROB_EMA * self.prob_ema + (1.0 - PROB_EMA) * probs
                    probs_out = self.prob_ema.copy()

        # [FER#2] 얼굴 없는 프레임이 N개 이상 지속되면 EMA 리셋 (이전 사람 감정 잔류 방지)
        if probs_out is None:
            self._no_face_count += 1
            if self._no_face_count >= NO_FACE_RESET_FRAMES and self.prob_ema is not None:
                _logger.debug(
                    "[FER] No face for %d frames — resetting prob_ema", self._no_face_count
                )
                self.prob_ema = None
            if self._no_face_count >= NO_FACE_RESET_FRAMES:
                self._eye_closed_since = None
                self._is_drowsy = False

            # [FER#5] 항상 dict 반환으로 통일 (None 반환 제거)
            return {"packet": None, "roi_preview": None, "face_detected": False}

        argmax_idx   = int(np.argmax(probs_out))
        argmax_label = LABELS[argmax_idx]
        latency_ms   = (time.time() - frame_ts) * 1000.0
        self._latency_samples.append(latency_ms)

        packet = FERPacket(
            ts=frame_ts,  # [FER#6] time.time() 기반
            seq=self.seq,
            class_order=LABELS,
            softmax=[float(x) for x in probs_out],
            argmax_idx=argmax_idx,
            argmax_label=argmax_label,
            face_detected=face_detected,
            latency_ms=float(latency_ms),
            ear=float(ear_value) if ear_value is not None else None,
            is_drowsy=self._is_drowsy,
        )
        self.seq += 1

        if self.publisher is not None:
            self.publisher.publish(packet)

        packet_dict = packet.to_dict()

        # [FER#5] return_debug 여부와 무관하게 항상 동일한 구조의 dict 반환
        if return_debug:
            return {"packet": packet_dict, "roi_preview": roi_preview, "face_detected": face_detected}
        return {"packet": packet_dict, "roi_preview": None, "face_detected": face_detected}

    def get_metrics(self) -> dict:
        samples = self._latency_samples
        if samples:
            avg_lat = sum(samples) / len(samples)
            min_lat = min(samples)
            max_lat = max(samples)
        else:
            avg_lat = min_lat = max_lat = 0.0
        return {
            "fer_frame_count": self.seq,
            "avg_latency_ms": round(avg_lat, 2),
            "min_latency_ms": round(min_lat, 2),
            "max_latency_ms": round(max_lat, 2),
        }

    def close(self):
        self.face_mesh.close()
