# sensor_types.py
from dataclasses import dataclass
from typing import Dict


@dataclass
class SensorResult:
    """
    생체신호(BIO) 모델의 출력 결과를 담는 자료형.
    - raw_metrics   : hrv, gsr 등 원본값
    - emotion_scores: "calm", "stressed" 확률
    """
    raw_metrics: Dict[str, float]
    emotion_scores: Dict[str, float]
