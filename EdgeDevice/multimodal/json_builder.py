# json_builder.py
import json
from .multimodal_types import FusionResult


def fusion_result_to_json(result: FusionResult) -> str:
    """
    FusionResult를 보고서에서 정의한 JSON 형태로 변환.
    """
    data = {
        "timestamp": result.timestamp,
        "fusion": {
            "safe": result.fused_scores.get("safe", 0.0),
            "stressed": result.fused_scores.get("stressed", 0.0),
            "dominant": result.dominant_emotion,
            "confidence": result.confidence,
        },
        "fer": {
            "emotion_scores": result.fer.emotion_scores,
            "drowsy_scores": result.fer.drowsy_scores,
        },
        "sensor": {
            "raw_metrics": result.sensor.raw_metrics,
            "emotion_scores": result.sensor.emotion_scores,
        },
        "alert": {
            "level": result.alert_level,
            "reason": result.alert_reason,
        },
    }
    return json.dumps(data, ensure_ascii=False)
