# multimodal_engine.py
import time
from typing import Dict, Optional

from .fer_interface import FerSource
from .sensor_interface import SensorSource
from .fer_types import FerResult
from .sensor_types import SensorResult
from .multimodal_types import FusionResult


class MultiModalEngine:
    """
    멀티모달 기반 운전자 상태 분석 엔진 (테스트용 수정 버전)
    """

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

        if abs(self.w_fer + self.w_sensor - 1.0) > 1e-6:
            raise ValueError(
                f"[엔진 설정 오류] w_fer({self.w_fer})와 w_sensor({self.w_sensor})의 합은 "
                f"반드시 1.0이어야 합니다! (현재 합: {self.w_fer + self.w_sensor:.2f})"
            )
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
        safe     = emo.get("safe", 0.0)
        stressed = emo.get("stressed", 0.0)
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _fuse_scores(
        self,
        fer_scores: Dict[str, float],
        sensor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        safe = (
            self.w_fer * fer_scores["safe"] + self.w_sensor * sensor_scores["safe"]
        )
        stressed = (
            self.w_fer * fer_scores["stressed"] + self.w_sensor * sensor_scores["stressed"]
        )
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _smooth_stress(self, current_stress: float) -> float:
        if self.smoothed_stress is None:
            self.smoothed_stress = current_stress
        else:
            a = self.ema_alpha
            self.smoothed_stress = a * current_stress + (1.0 - a) * self.smoothed_stress
        return self.smoothed_stress

    def _decide_alert(
        self,
        ema_stress: float,
        fer: Optional[FerResult],
    ) -> tuple[str, str]:
        drowsy_val = fer.drowsy_scores.get("drowsy", 0.0)if fer else 0.0
        face_missing_val = fer.drowsy_scores.get("face_missing", 0.0)if fer else 0.0

        if face_missing_val == 1.0:
            # 위험(DANGER)이 아닌 경고(WARNING) 수준으로 처리하여 오작동 방지
            level = "WARNING" 
            reason = "Face not detected. Please look at the camera."

        elif ema_stress >= 0.75 or drowsy_val >= 0.60:
            level = "DANGER"
            reason = f"High stress (EMA={ema_stress:.2f}) or drowsiness ({drowsy_val:.2f})"
        elif ema_stress >= 0.50:
            level = "WARNING"
            reason = f"Stress level is elevated (EMA={ema_stress:.2f})"
        else:
            level = "NORMAL"
            reason = f"Stress level is low (EMA={ema_stress:.2f})"

        return level, reason

    def step(self) -> Optional[FusionResult]:
        fer_res    = self.fer_source.get_latest_result()
        sensor_res = self.sensor_source.get_latest_result()

        # 1.둘 다 없을 때만(완전한 먹통 상태) 엔진을 멈춥니다. (기존 or 를 and 로 변경!)
        if fer_res is None and sensor_res is None:
            return None

        # 2. 데이터 캐싱 (있는 것만 업데이트)
        if fer_res: self.last_fer = fer_res
        if sensor_res: self.last_sensor = sensor_res

        # 3. 각 모달리티별 점수 변환 (데이터가 없으면 None 할당)
        fer_safe_stress = self._convert_fer_to_safe_stress(fer_res) if fer_res else None
        sensor_safe_stress = self._convert_sensor_to_safe_stress(sensor_res) if sensor_res else None

        # 4.Degraded Mode (결함 허용) 퓨전 로직
        confidence_penalty = 1.0 # 기본값 (둘 다 있을 땐 100% 신뢰)
        
        if fer_res and sensor_res:
            # 상태 A: 완벽한 퓨전 (둘 다 정상 작동)
            fused_scores = self._fuse_scores(fer_safe_stress, sensor_safe_stress)
        elif fer_res and not sensor_res:
            # 상태 B: FER만 작동 (BIO 워밍업 중이거나 연결 끊김)
            fused_scores = fer_safe_stress
            confidence_penalty = 0.7 # 센서가 없으므로 확신도를 30% 깎음 (원하는 수치로 조절 가능)
        elif not fer_res and sensor_res:
            # 상태 C: BIO만 작동 (카메라 로드 실패 등)
            fused_scores = sensor_safe_stress
            confidence_penalty = 0.7 # 카메라가 없으므로 확신도를 30% 깎음


        # 여기서 EMA 계산을 여기서 수행함
        current_raw_stress = fused_scores["stressed"]
        ema_stress = self._smooth_stress(current_raw_stress)

        # 5. 우세한 감정과 최종 신뢰도 계산
        dominant = "stressed" if fused_scores["stressed"] >= fused_scores["safe"] else "safe"
        confidence = fused_scores[dominant] * confidence_penalty # 페널티 적용!

        # 6. 알림 판별
        alert_level, alert_reason = self._decide_alert(ema_stress, fer_res)


        if fer_res and sensor_res:
            # 상태 A: 둘 다 살아있으면, 둘 중 더 오래된(보수적인) 시간을 쓴다
            actual_observation_time = min(fer_res.timestamp, sensor_res.timestamp)
        elif fer_res and not sensor_res:
            # 상태 B: 센서가 죽었으면, 살아있는 카메라 시간만 쓴다
            actual_observation_time = fer_res.timestamp
        elif not fer_res and sensor_res:
            # 상태 C: 카메라가 죽었으면, 살아있는 센서 시간만 쓴다
            actual_observation_time = sensor_res.timestamp
        else:
            # 상태 D: 둘 다 죽었을 때 
            actual_observation_time = time.time()

        return FusionResult(
            timestamp=actual_observation_time,
            fused_scores=fused_scores,
            dominant_emotion=dominant,
            confidence=confidence,
            fer=fer_res,        # Degraded Mode에서는 둘 중 하나가 None일 수 있음
            sensor=sensor_res,  # Degraded Mode에서는 둘 중 하나가 None일 수 있음
            alert_level=alert_level,
            alert_reason=alert_reason,
        )