import logging
import time
from typing import Dict, Optional, Tuple  # [MUL#10] Tuple: Python 3.8 호환

from .fer_interface import FerSource
from .sensor_interface import SensorSource
from .fer_types import FerResult
from .sensor_types import SensorResult
from .multimodal_types import FusionResult

_logger = logging.getLogger(__name__)


class MultiModalEngine:
    def __init__(
        self,
        fer_source: FerSource,
        sensor_source: SensorSource,
        w_fer: float = 0.4,
        w_sensor: float = 0.6,
        ema_alpha: float = 0.3,
    ) -> None:
        self.fer_source    = fer_source
        self.sensor_source = sensor_source
        self.ema_alpha     = ema_alpha
        self.smoothed_stress: Optional[float] = None
        self.last_fer:    Optional[FerResult]    = None
        self.last_sensor: Optional[SensorResult] = None

        # [MUL#6] w_fer + w_sensor != 1.0 시 자동 정규화 후 경고
        total = w_fer + w_sensor
        if abs(total - 1.0) > 1e-6:
            _logger.warning(
                "[MultiModal] w_fer(%.3f) + w_sensor(%.3f) = %.4f ≠ 1.0, normalizing.",
                w_fer, w_sensor, total,
            )
            w_fer    /= total
            w_sensor /= total
        self.w_fer    = w_fer
        self.w_sensor = w_sensor

    # ------------------------------------------------------------------
    # 내부 유틸리티
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_pair(values: Dict[str, float]) -> Dict[str, float]:
        safe     = values.get("safe", 0.0)
        stressed = values.get("stressed", 0.0)
        total    = safe + stressed
        if total <= 1e-8:
            return {"safe": 1.0, "stressed": 0.0}
        return {"safe": safe / total, "stressed": stressed / total}

    @staticmethod
    def _make_neutral_fer() -> FerResult:
        """[MUL#2] BIO-only 모드에서 사용할 중립 FerResult (FER 미가용 시)."""
        return FerResult(
            emotion_scores={"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0},
            drowsy_scores={"alert": 0.5, "drowsy": 0.5},
            ts=0.0,
        )

    @staticmethod
    def _make_neutral_sensor() -> SensorResult:
        """[MUL#2] FER-only 모드에서 사용할 중립 SensorResult (BIO 미가용 시)."""
        return SensorResult(
            raw_metrics={"bpm": 0.0, "hrv": 0.0, "gsr": 0.0, "bio_conf": 0.0},
            emotion_scores={"safe": 0.5, "stressed": 0.5},
            ts=0.0,
        )

    # ------------------------------------------------------------------
    # 변환 / 퓨전
    # ------------------------------------------------------------------

    def _convert_fer_to_safe_stress(self, fer: FerResult) -> Dict[str, float]:
        emo = fer.emotion_scores

        happy   = emo.get("happy",   0.0)
        neutral = emo.get("neutral", 0.0)
        sad     = emo.get("sad",     0.0)
        angry   = emo.get("angry",   0.0)

        # [MUL#9] drowsy 이진값 의존 제거: 얼굴 유무 기반 drowsy는 잘못된 신호.
        # 순수 감정 점수만으로 safe/stressed 계산.
        # 실제 EAR 기반 졸음 감지 구현 시 drowsy 항 재추가 가능.
        stressed = sad + angry
        safe     = happy + neutral
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _convert_sensor_to_safe_stress(self, sensor: SensorResult) -> Dict[str, float]:
        emo      = sensor.emotion_scores
        safe     = emo.get("safe",     0.0)
        stressed = emo.get("stressed", 0.0)
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _fuse_scores(
        self,
        fer_scores:    Dict[str, float],
        sensor_scores: Dict[str, float],
    ) -> Dict[str, float]:
        safe     = self.w_fer * fer_scores["safe"]     + self.w_sensor * sensor_scores["safe"]
        stressed = self.w_fer * fer_scores["stressed"] + self.w_sensor * sensor_scores["stressed"]
        return self._normalize_pair({"safe": safe, "stressed": stressed})

    def _smooth_stress(self, current_stress: float) -> float:
        if self.smoothed_stress is None:
            self.smoothed_stress = current_stress
        else:
            a = self.ema_alpha
            self.smoothed_stress = a * current_stress + (1.0 - a) * self.smoothed_stress
        return self.smoothed_stress

    # [MUL#3] _decide_alert는 이미 계산된 ema_stress를 인자로 받음.
    # 내부에서 self.smoothed_stress를 변경하는 side effect 제거.
    def _decide_alert(self, ema_stress: float) -> Tuple[str, str]:  # [MUL#10]
        # [MUL#1] drowsy_val >= 0.60 DANGER 트리거 제거.
        # 얼굴 유무로만 판단하던 drowsy가 고개 돌림/조명 문제 시 DANGER를 유발했음.
        if ema_stress >= 0.75:
            level  = "DANGER"
            reason = f"High stress (EMA={ema_stress:.2f})"
        elif ema_stress >= 0.50:
            level  = "WARNING"
            reason = f"Stress level is elevated (EMA={ema_stress:.2f})"
        else:
            level  = "NORMAL"
            reason = f"Stress level is low (EMA={ema_stress:.2f})"
        return level, reason

    # ------------------------------------------------------------------
    # 메인 스텝
    # ------------------------------------------------------------------

    def step(self) -> Optional[FusionResult]:
        fer_res    = self.fer_source.get_latest_result()
        sensor_res = self.sensor_source.get_latest_result()

        # [MUL#2] Degraded Mode: 한쪽 모달리티만 있어도 동작
        if fer_res is None and sensor_res is None:
            return None

        degraded = False
        if sensor_res is None:
            # FER-only 모드: BIO가 아직 준비되지 않았거나 연결 실패
            sensor_res = self._make_neutral_sensor()
            degraded   = True
            _logger.debug("[MultiModal] Degraded mode: BIO unavailable, using neutral sensor.")
        elif fer_res is None:
            # BIO-only 모드: FER 모델 로드 실패 또는 카메라 없음
            fer_res  = self._make_neutral_fer()
            degraded = True
            _logger.debug("[MultiModal] Degraded mode: FER unavailable, using neutral FER.")

        self.last_fer    = fer_res
        self.last_sensor = sensor_res

        fer_safe_stress    = self._convert_fer_to_safe_stress(fer_res)
        sensor_safe_stress = self._convert_sensor_to_safe_stress(sensor_res)

        fused_scores = self._fuse_scores(fer_safe_stress, sensor_safe_stress)
        dominant     = "stressed" if fused_scores["stressed"] >= fused_scores["safe"] else "safe"

        # [MUL#7] confidence = fused 확률 × 신호 품질 반영
        # BIO 신호 품질(bio_conf)과 FER 최고 확률 모두를 고려
        bio_conf    = sensor_res.raw_metrics.get("bio_conf", 1.0)
        fer_quality = max(fer_res.emotion_scores.values()) if fer_res.emotion_scores else 1.0
        if degraded:
            signal_quality = 0.5  # 단일 모달리티: 품질 50%로 제한
        else:
            signal_quality = min(bio_conf, fer_quality)
        confidence = fused_scores[dominant] * signal_quality

        # [MUL#3] _smooth_stress를 step()에서 명시적으로 호출 후 결과를 _decide_alert에 전달
        ema_stress              = self._smooth_stress(fused_scores["stressed"])
        alert_level, alert_reason = self._decide_alert(ema_stress)

        # [MUL#5] FusionResult 타임스탬프: step() 완료 시점이 아닌 실제 관측 시점 기준
        # 두 소스 중 더 오래된 타임스탬프를 사용 (BIO는 최대 1초 전 데이터일 수 있음)
        fer_ts    = fer_res.ts    if fer_res.ts    > 0 else time.time()
        sensor_ts = sensor_res.ts if sensor_res.ts > 0 else time.time()
        obs_timestamp = min(fer_ts, sensor_ts)

        return FusionResult(
            timestamp=obs_timestamp,
            fused_scores=fused_scores,
            dominant_emotion=dominant,
            confidence=confidence,
            fer=fer_res,
            sensor=sensor_res,
            alert_level=alert_level,
            alert_reason=alert_reason,
        )
