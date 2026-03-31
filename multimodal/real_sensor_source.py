# real_sensor_source.py
import time
from typing import Dict, Optional

from .sensor_interface import SensorSource
from .sensor_types import SensorResult


class RealSensorSource(SensorSource):
    def __init__(self) -> None:
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        # 1. BIO 코어에서 계산한 스트레스 수치
        raw_stress = float(payload.get("bio_arousal", 0.0))

        # 2. 센서 접촉 신뢰도 (손가락이 잘 붙어있는가?)
        bio_conf = float(payload.get("bio_conf", 1.0))

        # 3. 신뢰도가 낮으면 0.5(판단 보류)로 수렴
        #    신뢰도 1.0 → raw_stress 그대로 / 신뢰도 0.0 → 중립(0.5)
        stressed = (raw_stress * bio_conf) + (0.5 * (1.0 - bio_conf))
        safe = ((1.0 - raw_stress) * bio_conf) + (0.5 * (1.0 - bio_conf))

        raw_metrics: Dict[str, float] = {
            "bpm": float(payload.get("bpm") or 0.0),
            "hrv": float(payload.get("rmssd_use") or 0.0),
            "gsr": float(payload.get("gsr_cur") or 0.0),
        }

        emotion_scores: Dict[str, float] = {
            "safe": round(safe, 4),
            "stressed": round(stressed, 4),
        }

        obs_time = float(payload.get("timestamp_unix", time.time()))
        self._latest = SensorResult(
            timestamp=obs_time,
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
            bio_conf=bio_conf,  # [QUALITY #3 연동] confidence 보정을 위해 저장
        )

    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest