# fer_types.py
from dataclasses import dataclass
from typing import Dict


@dataclass
class FerResult:
    """
    FER 모델의 출력 결과를 담는 자료형.
    - emotion_scores: Angry / Sad / Happy / Neutral 확률
    - drowsy_scores : Alert / Drowsy 확률
    """
    emotion_scores: Dict[str, float]
    drowsy_scores: Dict[str, float]
