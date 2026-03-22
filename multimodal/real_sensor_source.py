# real_sensor_source.py
from typing import Dict, Optional

from sensor_interface import SensorSource
from sensor_types import SensorResult


class RealSensorSource(SensorSource):
    """
    찬희(BIO)가 update_window()로 payload를 넘겨주면
    SensorResult로 변환해서 멀티모달 엔진에 전달.

    찬희 payload 포맷:
    {
        "timestamp": "...",
        "bio_arousal": 0.7,   ← 스트레스 지수 (0~1)
        "bio_conf": 0.9,      ← 신뢰도
        "bpm": 88.0,          ← 심박수
        "rmssd_use": 0.05,    ← HRV
        "gsr_cur": 150.0,     ← 피부전도도
        ...
    }
    """

    def __init__(self) -> None:
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        """찬희가 호출해주는 함수 - payload 받아서 SensorResult로 변환"""

        # bio_arousal → stressed 로 매핑 (0~1 스트레스 지수)
        bio_arousal = float(payload.get("bio_arousal", 0.0))
        bio_conf    = float(payload.get("bio_conf", 1.0))

        # 신뢰도 낮으면 스트레스 점수 낮춤
        stressed = bio_arousal * bio_conf
        safe     = 1.0 - stressed

        raw_metrics: Dict[str, float] = {
            "bpm": float(payload.get("bpm") or 0.0),
            "hrv": float(payload.get("rmssd_use") or 0.0),
            "gsr": float(payload.get("gsr_cur") or 0.0),
        }

        emotion_scores: Dict[str, float] = {
            "safe":     round(safe, 4),
            "stressed": round(stressed, 4),
        }

        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=emotion_scores,
        )

    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest


# =========================================================
# 단위 테스트
# =========================================================
if __name__ == "__main__":
    print("[단위 테스트 시작] RealSensorSource 찬희 payload 변환 검증")

    sensor_src = RealSensorSource()

    # 찬희가 보내주는 가짜 payload
    dummy_payload = {
        "timestamp": "2026-03-22 15:00:00",
        "bio_arousal": 0.8,
        "bio_conf": 0.9,
        "bpm": 95.0,
        "rmssd_use": 0.04,
        "gsr_cur": 200.0,
        "ppgQ": 0.85,
        "finger_on": True,
        "gsr_fresh": True,
        "note": "",
    }

    sensor_src.update_window(dummy_payload)
    result = sensor_src.get_latest_result()

    if result:
        print(f"\n[변환 결과]")
        print(f"  raw_metrics   : {result.raw_metrics}")
        print(f"  emotion_scores: {result.emotion_scores}")

        stressed = result.emotion_scores.get("stressed", 0)
        if stressed > 0:
            print("\n결과: PASS (찬희 payload 정상 변환!)")
        else:
            print("\n결과: FAIL")
    else:
        print("\n결과: FAIL (데이터 없음)")