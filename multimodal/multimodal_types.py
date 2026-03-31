# multimodal_types.py
from dataclasses import dataclass
from typing import Dict, Optional

from .fer_types import FerResult
from .sensor_types import SensorResult


@dataclass
class FusionResult:
    """
    멀티모달 엔진의 최종 출력 구조.

    수정 이력:
        - [BUG] fer, sensor 필드를 Optional로 변경.
          Degraded Mode에서 엔진이 fer=None 또는 sensor=None을 넣을 수 있으므로
          Non-optional 선언은 타입 에러(mypy/pyright)를 유발하고
          json_builder 등 하위 소비자에서 AttributeError 크래시로 이어짐.
    """
    timestamp: float
    fused_scores: Dict[str, float]      # {"safe": x, "stressed": y}
    dominant_emotion: str               # "safe" 또는 "stressed"
    confidence: float
    fer: Optional[FerResult]            # Degraded Mode(카메라 없음)에서 None 가능
    sensor: Optional[SensorResult]      # Degraded Mode(센서 없음)에서 None 가능
    alert_level: str                    # "NORMAL" / "WARNING" / "DANGER"
    alert_reason: Optional[str]