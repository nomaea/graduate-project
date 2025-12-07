# real_sensor_source.py
from typing import Dict, Optional

from sensor_interface import SensorSource
from sensor_types import SensorResult
from bio_service import BioService


class RealSensorSource(SensorSource):
    """
    실제 BIO 파이프라인(BioService)을 사용해서 SensorResult를 만들어주는 구현체.
    외부에서 생체신호 윈도우(payload)를 넣어주면,
    내부적으로 BioService.process_window()를 돌리고,
    그 결과를 SensorResult로 변환해서 보관한다.
    """

    def __init__(self) -> None:
        self._bio = BioService()
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        """
        payload 예시:
        {
            "timestamp": "2025-11-17T12:00:00Z",
            "user_id": "user-001",
            "hr_series": [...],
            "eda_series": [...],
            "acc_series": [...],
        }
        """
        bio_out = self._bio.process_window(payload)

        # BioService가 돌려주는 features 예시:
        # {
        #   "hr_mean": 93.0,
        #   "hr_std": ...,
        #   "eda_mean": 0.195,
        #   "eda_std": ...,
        #   "acc_mean": ...,
        #   "acc_std": ...,
        # }
        features: Dict[str, float] = bio_out.get("features", {})

        # 🔹 여기서 멀티모달 쪽이 기대하는 키 이름으로 매핑
        #   - hrv  ← hr_mean (심박 관련 값)
        #   - gsr  ← eda_mean (피부전도/스트레스 관련 값)
        raw_metrics: Dict[str, float] = {
            "hrv": float(features.get("hr_mean", 0.0)),
            "gsr": float(features.get("eda_mean", 0.0)),
        }

        # 멀티모달 엔진이 사용하는 sensor.emotion_scores
        emotion_scores: Dict[str, float] = {
            "calm": float(bio_out.get("bio_safe", 0.0)),
            "stressed": float(bio_out.get("bio_stress", 0.0)),
        }

        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
        )

    def get_latest_result(self) -> Optional[SensorResult]:
        """
        멀티모달 엔진이 주기적으로 호출하는 함수.
        아직 update_window가 한 번도 안 불렸으면 None을 반환할 수 있다.
        """
        return self._latest
