# mock/sensor_mock.py
import random
from typing import Optional, Dict

from sensor_interface import SensorSource
from sensor_types import SensorResult


class MockSensorSource(SensorSource):
    """
    실제 HRV/GSR 센서 없이도 엔진을 테스트하기 위한 Mock Sensor.
    """

    def get_latest_result(self) -> Optional[SensorResult]:
        # HRV: 40~100 사이 값, GSR: 0~1 정도로 가정
        hrv = random.uniform(40.0, 100.0)
        gsr = random.uniform(0.1, 1.0)

        raw_metrics: Dict[str, float] = {
            "hrv": hrv,
            "gsr": gsr,
        }

        # 센서 모델이 출력한 calm / stressed 확률을 가정
        calm = random.uniform(0.0, 1.0)
        stressed = random.uniform(0.0, 1.0)
        total = calm + stressed + 1e-8

        emotion_scores: Dict[str, float] = {
            "calm": calm / total,
            "stressed": stressed / total,
        }

        return SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
        )
