import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from fer_api import FERPacket, FERQueuePublisher


LABELS = ["neutral", "happy", "sad", "angry"]
IMG_SIZE = 224

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PROB_EMA = 0.65


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


def preprocess_roi_bgr(roi_bgr: np.ndarray):
    roi_resized = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    x = roi_rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.expand_dims(x, axis=0)  # NHWC
    return x, roi_resized


def get_mesh_aligned_roi(frame_bgr: np.ndarray, face_landmarks, scale=1.45):
    fh, fw = frame_bgr.shape[:2]

    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    pts = []
    for lm in face_landmarks.landmark:
        x = lm.x * fw
        y = lm.y * fh
        pts.append((x, y))
    pts = np.array(pts, dtype=np.float32)

    lx, ly = pts[LEFT_EYE_OUTER]
    rx, ry = pts[RIGHT_EYE_OUTER]

    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    eye_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    rotated, rot_mat = rotate_image(frame_bgr, angle, eye_center)
    rot_pts = np.array([transform_point(x, y, rot_mat) for x, y in pts], dtype=np.float32)

    min_x = float(np.min(rot_pts[:, 0]))
    max_x = float(np.max(rot_pts[:, 0]))
    min_y = float(np.min(rot_pts[:, 1]))
    max_y = float(np.max(rot_pts[:, 1]))

    x1, y1, x2, y2 = make_square_box(min_x, min_y, max_x, max_y, fw, fh, scale=scale)
    if x2 <= x1 or y2 <= y1:
        return None

    roi_bgr = rotated[y1:y2, x1:x2]
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

        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
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
        frame_ts = time.monotonic()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        probs_out = None
        face_detected = False
        roi_preview = None

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            aligned_roi = get_mesh_aligned_roi(frame_bgr, face_landmarks, scale=1.45)

            if aligned_roi is not None and aligned_roi.size > 0:
                face_detected = True
                x_nhwc, roi_preview = preprocess_roi_bgr(aligned_roi)
                x_in = x_nhwc.astype(self.input_details["dtype"])

                self.interpreter.set_tensor(self.input_details["index"], x_in)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_details["index"])[0]
                probs = softmax(out)

                if self.prob_ema is None:
                    self.prob_ema = probs.copy()
                else:
                    self.prob_ema = PROB_EMA * self.prob_ema + (1.0 - PROB_EMA) * probs

                probs_out = self.prob_ema.copy()

        if probs_out is None:
            if return_debug:
                return {
                    "packet": None,
                    "roi_preview": None,
                    "face_detected": False,
                }
            return None

        argmax_idx = int(np.argmax(probs_out))
        argmax_label = LABELS[argmax_idx]
        latency_ms = (time.monotonic() - frame_ts) * 1000.0

        packet = FERPacket(
            ts=frame_ts,
            seq=self.seq,
            class_order=LABELS,
            softmax=[float(x) for x in probs_out],
            argmax_idx=argmax_idx,
            argmax_label=argmax_label,
            face_detected=face_detected,
            latency_ms=float(latency_ms),
        )
        self.seq += 1

        if self.publisher is not None:
            self.publisher.publish(packet)

        packet_dict = packet.to_dict()

        if return_debug:
            return {
                "packet": packet_dict,
                "roi_preview": roi_preview,
                "face_detected": face_detected,
            }

        return packet_dict

    def close(self):
        self.face_mesh.close()