# multimodal_types.py
from dataclasses import dataclass
from typing import Dict, Optional

from fer_types import FerResult
from sensor_types import SensorResult


@dataclass
class FusionResult:
    """
    멀티모달 엔진의 최종 출력 구조.
    """
    timestamp: float
    fused_scores: Dict[str, float]      # {"safe": x, "stressed": y}
    dominant_emotion: str              # "safe" 또는 "stressed"
    confidence: float
    fer: FerResult
    sensor: SensorResult
    alert_level: str                   # "NORMAL" / "WARNING" / "DANGER"
    alert_reason: Optional[str]
