# sensor_types.py
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SensorResult:
    """
    생체신호(BIO) 모델의 출력 결과를 담는 자료형.
    - raw_metrics   : hrv, gsr 등 원본값
    - emotion_scores: "safe", "stressed" 확률
    - bio_conf      : 센서 접촉 신뢰도 (0.0 ~ 1.0)

    수정 이력:
        - [QUALITY #3 연동] bio_conf 필드 추가.
          RealSensorSource.update_window()에서 이미 계산하고 있었지만
          SensorResult에 저장하지 않아 multimodal_engine의 confidence 보정에
          활용되지 못하고 있었음. 기본값 1.0으로 하위 호환성 유지.
    """
    timestamp: float
    raw_metrics: Dict[str, float]
    emotion_scores: Dict[str, float]
    bio_conf: float = field(default=1.0)    # 센서 접촉 신뢰도 (기본값: 완전 신뢰)