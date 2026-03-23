import time
from typing import Dict, Optional

from .fer_interface import FerSource
from .sensor_interface import SensorSource
from .fer_types import FerResult
from .sensor_types import SensorResult
from .multimodal_types import FusionResult


class MultiModalEngine:
    def __init__(
        self,
        fer_source: FerSource,
        sensor_source: SensorSource,
        w_fer: float = 0.4,
        w_sensor: float = 0.6,
        ema_alpha: float = 0.3,
    ) -> None:
        self.fer_source = fer_source
        self.sensor_source = sensor_source
        self.w_fer = w_fer
        self.w_sensor = w_sensor
        self.ema_alpha = ema_alpha
        self.smoothed_stress: Optional[float] = None
        self.last_fer: Optional[FerResult] = None
        self.last_sensor: Optional[SensorResult] = None

    @staticmethod
    def _normalize_pair(values: Dict[str, float]) -> Dict[str, float]:
        safe = values.get("safe", 0.0)
        stressed = values.get("stressed", 0.0)
        total = safe + stressed
        if total <= 1e-8:
            return {"safe": 1.0, "stressed": 0.0}
        return {"safe": safe / total, "stressed": stressed / total}

    def _convert_fer_to_safe_stress(self, fer: FerResult) -> Dict[str, float]:
        emo = fer.emotion_scores
        drowsy = fer.drowsy_scores

        happy = emo.get("happy", 0.0)
        neutral = emo.get("neutral", 0.0)
        sad = emo.get("sad", 0.0)
        angry = emo.get("angry", 0.0)

        alert = drowsy.get("alert", 0.0)
        drowsy_val = drowsy.get("drowsy", 0.0)

        stressed = 0.6 * (sad + angry) + 0.4 * drowsy_val
        safe = 0.6 * (happy + neutral) + 0.4 * alert
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _convert_sensor_to_safe_stress(self, sensor: SensorResult) -> Dict[str, float]:
        emo = sensor.emotion_scores
        safe = emo.get("safe", 0.0)
        stressed = emo.get("stressed", 0.0)
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _fuse_scores(
        self,
        fer_scores: Dict[str, float],
        sensor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        safe = self.w_fer * fer_scores["safe"] + self.w_sensor * sensor_scores["safe"]
        stressed = self.w_fer * fer_scores["stressed"] + self.w_sensor * sensor_scores["stressed"]
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _smooth_stress(self, current_stress: float) -> float:
        if self.smoothed_stress is None:
            self.smoothed_stress = current_stress
        else:
            a = self.ema_alpha
            self.smoothed_stress = a * current_stress + (1.0 - a) * self.smoothed_stress
        return self.smoothed_stress

    def _decide_alert(self, fused_scores: Dict[str, float], fer: FerResult) -> tuple[str, str]:
        raw_stressed = fused_scores["stressed"]
        ema_stress = self._smooth_stress(raw_stressed)
        drowsy_val = fer.drowsy_scores.get("drowsy", 0.0)

        if ema_stress >= 0.75 or drowsy_val >= 0.60:
            level = "DANGER"
            reason = f"High stress (EMA={ema_stress:.2f}) or elevated drowsiness (drowsy={drowsy_val:.2f})"
        elif ema_stress >= 0.50:
            level = "WARNING"
            reason = f"Stress level is elevated (EMA={ema_stress:.2f})"
        else:
            level = "NORMAL"
            reason = f"Stress level is low (EMA={ema_stress:.2f})"

        return level, reason

    def step(self) -> Optional[FusionResult]:
        fer_res = self.fer_source.get_latest_result()
        sensor_res = self.sensor_source.get_latest_result()

        # 1. 영상(FER) 데이터조차 없으면 아예 판단 불가 (대기)
        if fer_res is None:
            return None

        self.last_fer = fer_res
        fer_safe_stress = self._convert_fer_to_safe_stress(fer_res)

        # 2. 영상은 있는데 생체(BIO) 데이터가 아직 없는 경우 (초기 센서 분석 대기 중)
        if sensor_res is None:
            dominant = "stressed" if fer_safe_stress["stressed"] >= fer_safe_stress["safe"] else "safe"
            alert_level, alert_reason = self._decide_alert(fer_safe_stress, fer_res)
            
            # 임시 더미 센서 데이터 생성
            dummy_sensor = SensorResult(
                raw_metrics={"bpm": 0.0, "hrv": 0.0, "gsr": 0.0},
                emotion_scores={"safe": 0.0, "stressed": 0.0}
            )

            return FusionResult(
                timestamp=time.time(),
                fused_scores=fer_safe_stress,  # FER 데이터만 100% 반영
                dominant_emotion=dominant,
                confidence=fer_safe_stress[dominant],
                fer=fer_res,
                sensor=dummy_sensor,
                alert_level=alert_level,
                alert_reason=f"[FER Only] {alert_reason}"
            )

        # 3. 영상과 생체 데이터가 모두 준비된 경우 (정상 퓨전 진행)
        self.last_sensor = sensor_res
        sensor_safe_stress = self._convert_sensor_to_safe_stress(sensor_res)

        fused_scores = self._fuse_scores(fer_safe_stress, sensor_safe_stress)
        dominant = "stressed" if fused_scores["stressed"] >= fused_scores["safe"] else "safe"
        confidence = fused_scores[dominant]

        alert_level, alert_reason = self._decide_alert(fused_scores, fer_res)

        return FusionResult(
            timestamp=time.time(),
            fused_scores=fused_scores,
            dominant_emotion=dominant,
            confidence=confidence,
            fer=fer_res,
            sensor=sensor_res,
            alert_level=alert_level,
            alert_reason=alert_reason,
        )

 
