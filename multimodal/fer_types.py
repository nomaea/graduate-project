# fer_types.py
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class FerResult:
    """
    FER 모델의 출력 결과를 담는 자료형.
    - emotion_scores: Angry / Sad / Happy / Neutral 확률
    - drowsy_scores : Alert / Drowsy 확률 (현재는 EAR 미구현으로 중립값 0.5/0.5 고정)
    - ts            : 관측 시점 Unix 타임스탬프 (time.time() 기준)  ← [MUL#5]
    """
    emotion_scores: Dict[str, float]
    drowsy_scores: Dict[str, float]
    ts: float = 0.0
