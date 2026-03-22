# mock/fer_mock.py
import random
from typing import Optional, Dict

from fer_interface import FerSource
from fer_types import FerResult


class MockFerSource(FerSource):
    """
    실제 카메라/모델 없이도 멀티모달 엔진을 테스트하기 위한 Mock FER.
    """

    def get_latest_result(self) -> Optional[FerResult]:
        # 랜덤 감정 분포 생성 (softmax 느낌 내기 위해)
        angry = random.uniform(0.0, 1.0)
        sad = random.uniform(0.0, 1.0)
        happy = random.uniform(0.0, 1.0)
        neutral = random.uniform(0.0, 1.0)
        total_emo = angry + sad + happy + neutral + 1e-8

        emotion_scores: Dict[str, float] = {
            "angry": angry / total_emo,
            "sad": sad / total_emo,
            "happy": happy / total_emo,
            "neutral": neutral / total_emo,
        }

        # 졸음 분포
        alert = random.uniform(0.0, 1.0)
        drowsy = random.uniform(0.0, 1.0)
        total_d = alert + drowsy + 1e-8

        drowsy_scores: Dict[str, float] = {
            "alert": alert / total_d,
            "drowsy": drowsy / total_d,
        }

        return FerResult(
            emotion_scores=emotion_scores,
            drowsy_scores=drowsy_scores,
        )
