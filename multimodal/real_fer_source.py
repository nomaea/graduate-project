from typing import Optional, Dict
from queue import Queue, Empty

from .fer_interface import FerSource
from .fer_types import FerResult


LABEL_ORDER = ["neutral", "happy", "sad", "angry"]


class RealFerSource(FerSource):
    def __init__(self, fer_queue: Queue) -> None:
        self._fer_queue = fer_queue
        self._latest: Optional[FerResult] = None

    def get_latest_result(self) -> Optional[FerResult]:
        try:
            while True:
                packet = self._fer_queue.get_nowait()
                self._latest = self._convert(packet)
        except Empty:
            pass
        return self._latest

    def _convert(self, packet: dict) -> FerResult:
        softmax = packet.get("softmax", [0.0, 0.0, 0.0, 0.0])

        emotion_scores: Dict[str, float] = {
            label: float(softmax[i]) if i < len(softmax) else 0.0
            for i, label in enumerate(LABEL_ORDER)
        }

        # [MUL#1] + [FER#7] 얼굴 유무로 drowsy를 판단하는 잘못된 로직 제거.
        # - face_detected=True  → alert=1.0 은 실제 졸음이 없음을 보장하지 않음
        # - face_detected=False → drowsy=1.0 은 고개 돌림/조명 부족도 DANGER 유발
        # 실제 EAR(Eye Aspect Ratio) 기반 졸음 감지 구현 전까지 중립값(0.5/0.5) 고정.
        # 기본값도 False로 수정 [FER#7]: 얼굴 없는 상황을 "있는 것"으로 오판 방지.
        drowsy_scores: Dict[str, float] = {"alert": 0.5, "drowsy": 0.5}

        ts = float(packet.get("ts", 0.0))  # [MUL#5] 관측 시점 전달

        return FerResult(
            emotion_scores=emotion_scores,
            drowsy_scores=drowsy_scores,
            ts=ts,
        )
