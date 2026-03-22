from typing import Dict, Optional

from .sensor_interface import SensorSource
from .sensor_types import SensorResult


class RealSensorSource(SensorSource):
    def __init__(self) -> None:
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        bio_arousal = float(payload.get("bio_arousal", 0.0))
        bio_conf = float(payload.get("bio_conf", 1.0))

        stressed = bio_arousal * bio_conf
        safe = 1.0 - stressed

        raw_metrics: Dict[str, float] = {
            "bpm": float(payload.get("bpm") or 0.0),
            "hrv": float(payload.get("rmssd_use") or 0.0),
            "gsr": float(payload.get("gsr_cur") or 0.0),
        }

        emotion_scores: Dict[str, float] = {
            "safe": round(safe, 4),
            "stressed": round(stressed, 4),
        }

        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
        )

    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest
