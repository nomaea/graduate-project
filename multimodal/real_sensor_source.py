import time
from typing import Dict, Optional

from .sensor_interface import SensorSource
from .sensor_types import SensorResult

# [MUL#4] bio_conf 이 임계값 미만이면 신호를 신뢰할 수 없으므로 중립(0.5/0.5) 처리
_LOW_CONF_THRESHOLD = 0.3


class RealSensorSource(SensorSource):
    def __init__(self) -> None:
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        bio_arousal = float(payload.get("bio_arousal", 0.0))
        bio_conf    = float(payload.get("bio_conf", 1.0))

        # [MUL#4] 낮은 신호 품질(손가락 미접촉, 센서 미연결 등)을 "안전"으로 오해석 방지.
        # bio_conf가 낮을 때: stressed = bio_arousal * bio_conf ≈ 0 → safe ≈ 1.0 (오류)
        # 수정: 신뢰 불가 시 중립(0.5/0.5)으로 표현 → fusion 가중치 영향 최소화
        if bio_conf < _LOW_CONF_THRESHOLD:
            stressed = 0.5
            safe     = 0.5
        else:
            stressed = bio_arousal * bio_conf
            safe     = 1.0 - stressed

        raw_metrics: Dict[str, float] = {
            "bpm":      float(payload.get("bpm") or 0.0),
            "hrv":      float(payload.get("rmssd_use") or 0.0),
            "gsr":      float(payload.get("gsr_cur") or 0.0),
            "bio_conf": bio_conf,  # [MUL#7] confidence 품질 추적을 위해 전달
        }

        emotion_scores: Dict[str, float] = {
            "safe":     round(safe, 4),
            "stressed": round(stressed, 4),
        }

        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
            ts=time.time(),  # [MUL#5] 관측 시점 기록
        )

    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest
