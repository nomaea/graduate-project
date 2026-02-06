# real_fer_source.py

from typing import Optional, Dict, Any

import numpy as np
import cv2


# 맥북(tensorflow-macos)과 라즈베리파이(tflite-runtime) 모두 호환되는 코드
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    from tflite_runtime.interpreter import Interpreter

from fer_interface import FerSource
from fer_types import FerResult

MODEL_PATH = "models/multihead_fer_drowsy.tflite"
EMOTION_LABELS = ["angry", "happy", "neutral", "sad"]
DROWSY_LABELS  = ["alert", "drowsy"]


def load_interpreter():
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_frame(frame, input_details):
    input_shape = input_details["shape"]
    dtype = input_details["dtype"]

    if len(input_shape) != 4:
        raise ValueError(f"지원하지 않는 입력 shape: {input_shape}")

    b, d1, d2, d3 = input_shape
    if d1 in (1, 3) and d3 >= 16:
        data_format = "NCHW"
        c, h, w = int(d1), int(d2), int(d3)
    else:
        data_format = "NHWC"
        h, w, c = int(d1), int(d2), int(d3)

    if c == 1:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (w, h))
        if data_format == "NHWC":
            input_data = resized.reshape(1, h, w, 1)
        else:
            input_data = resized.reshape(1, 1, h, w)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (w, h))
        if data_format == "NHWC":
            input_data = resized.reshape(1, h, w, c)
        else:
            chw = np.transpose(resized, (2, 0, 1))
            input_data = chw.reshape(1, c, h, w)

    if dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(dtype)

    return input_data


def decode_outputs(outputs):
    """기존 코드 기반: 각 head에서 argmax만 뽑는 버전"""
    emotion_pred = None
    drowsy_pred = None

    for out in outputs:
        logits = out[0]
        num_classes = logits.shape[-1]

        idx = int(np.argmax(logits))
        conf = float(np.max(logits))

        if num_classes == len(EMOTION_LABELS):
            emotion_pred = (idx, conf)
        elif num_classes == len(DROWSY_LABELS):
            drowsy_pred = (idx, conf)

    return emotion_pred, drowsy_pred


class RealFerSource(FerSource):
    """
    TFLite FER + Drowsy 모델을 사용해서 FerResult를 만들어주는 구현체.
    외부에서 frame을 넘겨주면 update_frame()에서 추론하고,
    get_latest_result()로 멀티모달 엔진이 가져가도록 한다.
    """

    def __init__(self) -> None:
        self.interpreter = load_interpreter()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._latest: Optional[FerResult] = None

    def update_frame(self, frame: np.ndarray) -> None:
        """웹캠/영상 프레임 한 장을 받아 FER 추론 후 FerResult 저장"""

        # 1) 얼굴 검출 (가장 큰 얼굴 선택)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            face_roi = frame[y:y + h, x:x + w]
        else:
            face_roi = frame  # 얼굴 없으면 전체 프레임

        # 2) TFLite 추론
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_data = preprocess_frame(face_roi, input_details[0])
        self.interpreter.set_tensor(input_details[0]["index"], input_data)
        self.interpreter.invoke()

        outputs = []
        for od in output_details:
            outputs.append(self.interpreter.get_tensor(od["index"]))

        emotion_pred, drowsy_pred = decode_outputs(outputs)

        # 3) FerResult로 매핑 (가장 단순: top-1만 1.0, 나머지 0.0)
        emotion_scores: Dict[str, float] = {k: 0.0 for k in EMOTION_LABELS}
        drowsy_scores: Dict[str, float] = {k: 0.0 for k in DROWSY_LABELS}

        if emotion_pred is not None:
            emo_idx, emo_conf = emotion_pred
            if 0 <= emo_idx < len(EMOTION_LABELS):
                emotion_scores[EMOTION_LABELS[emo_idx]] = float(emo_conf)  # or 1.0

        if drowsy_pred is not None:
            d_idx, d_conf = drowsy_pred
            if 0 <= d_idx < len(DROWSY_LABELS):
                drowsy_scores[DROWSY_LABELS[d_idx]] = float(d_conf)  # or 1.0

        self._latest = FerResult(
            emotion_scores=emotion_scores,
            drowsy_scores=drowsy_scores,
        )

    def get_latest_result(self) -> Optional[FerResult]:
        return self._latest
