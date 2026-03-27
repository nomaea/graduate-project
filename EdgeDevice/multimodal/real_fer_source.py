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

        face_detected = packet.get("face_detected", True)
        drowsy_scores: Dict[str, float] = {
            "alert": 1.0 if face_detected else 0.0,
            "drowsy": 0.0 if face_detected else 1.0,
        }

        return FerResult(
            emotion_scores=emotion_scores,
            drowsy_scores=drowsy_scores,
        )
