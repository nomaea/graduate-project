# sensor_types.py
from dataclasses import dataclass
from typing import Dict


@dataclass
class SensorResult:
    """
    생체신호(BIO) 모델의 출력 결과를 담는 자료형.
    - raw_metrics   : bpm, hrv, gsr, bio_conf 등 원본값
    - emotion_scores: "safe" / "stressed" 확률
    - ts            : 관측 시점 Unix 타임스탬프 (time.time() 기준)  ← [MUL#5]
    """
    raw_metrics: Dict[str, float]
    emotion_scores: Dict[str, float]
    ts: float = 0.0
