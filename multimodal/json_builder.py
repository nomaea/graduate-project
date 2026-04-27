# json_builder.py
import json
from typing import Any, Dict
from .multimodal_types import FusionResult


def fusion_result_to_json(result: FusionResult) -> str:
    """
    FusionResult를 WebSocket 전송용 JSON 문자열로 변환.

    수정 이력:
        - [BUG] result.fer / result.sensor 가 Degraded Mode에서 None일 수 있음.
          기존 코드는 None 체크 없이 .emotion_scores 등에 직접 접근하여
          AttributeError 크래시 발생. None일 경우 빈 딕셔너리로 안전하게 대체.
    """
    # fer 블록: 카메라 없는 Degraded Mode에서는 None → 빈 딕셔너리
    if result.fer is not None:
        fer_block: Dict[str, Any] = {
            "emotion_scores": result.fer.emotion_scores,
            "drowsy_scores": result.fer.drowsy_scores,
        }
    else:
        fer_block = {}

    # sensor 블록: 센서 없는 Degraded Mode에서는 None → 빈 딕셔너리
    if result.sensor is not None:
        sensor_block: Dict[str, Any] = {
            "raw_metrics": result.sensor.raw_metrics,
            "emotion_scores": result.sensor.emotion_scores,
        }
    else:
        sensor_block = {}

    data = {
        "timestamp": result.timestamp,
        "fusion": {
            "safe": result.fused_scores.get("safe", 0.0),
            "stressed": result.fused_scores.get("stressed", 0.0),
            "dominant": result.dominant_emotion,
            "confidence": result.confidence,
        },
        "fer": fer_block,
        "sensor": sensor_block,
        "alert": {
            "level": result.alert_level,
            "reason": result.alert_reason,
        },
    }
    return json.dumps(data, ensure_ascii=False)