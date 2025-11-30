# multimodal_engine.py
import time
from typing import Dict, Optional

from fer_interface import FerSource
from sensor_interface import SensorSource
from fer_types import FerResult
from sensor_types import SensorResult
from multimodal_types import FusionResult


class MultiModalEngine:
    """
    멀티모달 기반 운전자 상태 분석 엔진.
    - FER 결과 + BIO 센서 결과를 safe/stressed 스케일로 변환
    - 가중치 기반 융합
    - EMA 스무딩
    - Alert 레벨 결정
    """

    def __init__(
        self,
        fer_source: FerSource,
        sensor_source: SensorSource,
        w_fer: float = 0.4,
        w_sensor: float = 0.6,
        ema_alpha: float = 0.3,
    ) -> None:
        # 입력 소스
        self.fer_source = fer_source
        self.sensor_source = sensor_source

        # 가중치 (합이 1이 되도록 권장)
        self.w_fer = w_fer
        self.w_sensor = w_sensor

        # EMA 계수
        self.ema_alpha = ema_alpha
        self.smoothed_stress: Optional[float] = None

        # 디버깅용 마지막 입력
        self.last_fer: Optional[FerResult] = None
        self.last_sensor: Optional[SensorResult] = None

    # --------------------------
    # 내부 헬퍼 함수들
    # --------------------------

    @staticmethod
    def _normalize_pair(values: Dict[str, float]) -> Dict[str, float]:
        """safe/stressed 두 값의 합을 1이 되도록 정규화."""
        safe = values.get("safe", 0.0)
        stressed = values.get("stressed", 0.0)
        total = safe + stressed
        if total <= 1e-8:
            # 둘 다 0이면 완전 safe로 가정
            return {"safe": 1.0, "stressed": 0.0}
        return {"safe": safe / total, "stressed": stressed / total}

    def _convert_fer_to_safe_stress(self, fer: FerResult) -> Dict[str, float]:
        """
        FER의 emotion_scores + drowsy_scores를 safe/stressed 스케일로 변환.
        보고서의 수식을 그대로 사용.
        """
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
        """
        SensorResult의 emotion_scores(calm, stressed)를 safe/stressed로 매핑.
        raw_metrics(HRV, GSR)는 융합 계산에는 직접 사용하지 않음.
        """
        emo = sensor.emotion_scores
        calm = emo.get("calm", 0.0)
        stressed = emo.get("stressed", 0.0)

        return self._normalize_pair({"safe": calm, "stressed": stressed})

    def _fuse_scores(
        self,
        fer_scores: Dict[str, float],
        sensor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        FER, Sensor 각각의 safe/stressed를 가중합으로 융합.
        """
        safe = (
            self.w_fer * fer_scores["safe"] + self.w_sensor * sensor_scores["safe"]
        )
        stressed = (
            self.w_fer * fer_scores["stressed"] + self.w_sensor * sensor_scores["stressed"]
        )

        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _smooth_stress(self, current_stress: float) -> float:
        """
        EMA(Exponential Moving Average)로 스트레스 값을 스무딩.
        """
        if self.smoothed_stress is None:
            self.smoothed_stress = current_stress
        else:
            a = self.ema_alpha
            self.smoothed_stress = a * current_stress + (1.0 - a) * self.smoothed_stress
        return self.smoothed_stress

    def _decide_alert(
        self,
        fused_scores: Dict[str, float],
        fer: FerResult,
    ) -> tuple[str, str]:
        """
        EMA 스무딩된 stress 값과 drowsy 값으로 Alert 레벨 결정.
        """
        raw_stressed = fused_scores["stressed"]
        ema_stress = self._smooth_stress(raw_stressed)
        drowsy_val = fer.drowsy_scores.get("drowsy", 0.0)

        # DANGER 조건
        if ema_stress >= 0.75 or drowsy_val >= 0.60:
            level = "DANGER"
            reason = (
                f"High stress (EMA={ema_stress:.2f}) "
                f"or elevated drowsiness (drowsy={drowsy_val:.2f})"
            )
        # WARNING 조건
        elif ema_stress >= 0.50:
            level = "WARNING"
            reason = f"Stress level is elevated (EMA={ema_stress:.2f})"
        # NORMAL
        else:
            level = "NORMAL"
            reason = f"Stress level is low (EMA={ema_stress:.2f})"

        return level, reason

    # --------------------------
    # 외부에서 호출하는 메인 함수
    # --------------------------

    def step(self) -> Optional[FusionResult]:
        """
        한 번 호출 시:
        1) FER 결과 수신
        2) Sensor 결과 수신
        3) safe/stressed 변환
        4) 가중치 기반 융합
        5) EMA + Alert 판단
        6) FusionResult 반환 (둘 중 하나라도 없으면 None)
        """
        fer_res = self.fer_source.get_latest_result()
        sensor_res = self.sensor_source.get_latest_result()

        if fer_res is None or sensor_res is None:
            return None

        self.last_fer = fer_res
        self.last_sensor = sensor_res

        fer_safe_stress = self._convert_fer_to_safe_stress(fer_res)
        sensor_safe_stress = self._convert_sensor_to_safe_stress(sensor_res)

        fused_scores = self._fuse_scores(fer_safe_stress, sensor_safe_stress)

        # dominant: safe 혹은 stressed 중 더 큰 값
        if fused_scores["stressed"] >= fused_scores["safe"]:
            dominant = "stressed"
        else:
            dominant = "safe"
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
