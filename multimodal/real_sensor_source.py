from typing import Dict, Optional

# 필요한 모듈 임포트
from .sensor_interface import SensorSource
from .sensor_types import SensorResult


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class RealSensorSource(SensorSource):
    """
    실제 PPG(심박) 및 GSR(피부전도도) 센서 데이터를 처리하여 
    멀티모달 엔진에 전달하는 구현체.
    """

    def __init__(self) -> None:
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        """
        보내줄 데이터 예시:
        {
            "timestamp": "...",
            "bio_arousal": 0.63,
            "bio_conf": 0.82,
            "bpm": 84.2,
            "rmssd_use": 0.041,
            "gsr_cur": 512.0,
            "gsr_slope_1s": 8.0,
            "gsr_peak_10s": 2,
            "ppgQ": 0.91,
            "finger_on": True,
            "gsr_fresh": True,
            "note": "..."
        }
        """
        # 1. BioService 실행 (에러 방지용 가짜/진짜 모델 사용)
        # 기존에는 BioService를 태웠지만,
        # 현재는 라즈베리파이에서 실행 중인 바이오 엔진이
        # 이미 최종 특징값과 추론값을 계산해서 payload로 넘겨주므로
        # 여기서는 payload 값을 직접 사용합니다.

        # 2. 데이터 매핑 (수정됨: 친구가 준 값을 최우선으로!)
        # 친구가 바이오 엔진 payload 키로 값을 줬다면,
        # 해당 값을 그대로 사용합니다.

        bpm = self._to_float(payload.get("bpm"), 0.0)
        rmssd_use = self._to_float(payload.get("rmssd_use"), 0.0)
        gsr_cur = self._to_float(payload.get("gsr_cur"), 0.0)
        bio_arousal = self._to_float(payload.get("bio_arousal"), 0.0)
        bio_conf = self._to_float(payload.get("bio_conf"), 0.0)
        gsr_slope_1s = self._to_float(payload.get("gsr_slope_1s"), 0.0)
        gsr_peak_10s = self._to_float(payload.get("gsr_peak_10s"), 0.0)
        ppg_q = self._to_float(payload.get("ppgQ"), 0.0)

        finger_on = bool(payload.get("finger_on", False))
        gsr_fresh = bool(payload.get("gsr_fresh", False))
        note = str(payload.get("note", ""))

        # 3. 멀티모달 엔진용 데이터 구조 생성
        raw_metrics: Dict[str, float] = {
            "bpm": bpm,
            "hrv": rmssd_use,
            "gsr": gsr_cur,
            "bio_arousal": bio_arousal,
            "bio_conf": bio_conf,
            "gsr_slope_1s": gsr_slope_1s,
            "gsr_peak_10s": gsr_peak_10s,
            "ppgQ": ppg_q,
            "finger_on": 1.0 if finger_on else 0.0,
            "gsr_fresh": 1.0 if gsr_fresh else 0.0,
        }

        # 4. 점수 계산
        sensor_emotion_scores = self._calculate_sensor_scores(bio_arousal)

        # 5. 결과 저장 (timestamp 제거 버전)
        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=sensor_emotion_scores
        )

    def _calculate_sensor_scores(self, bio_arousal: float) -> Dict[str, float]:
        """간단한 스트레스 점수 계산 로직"""
        stress_score = clamp01(bio_arousal)

        if stress_score < 0.1:
            stress_score = 0.0

        return {
            "safe": 1.0 - stress_score,
            "stressed": stress_score
        }

    # 부모 클래스가 요구하는 함수 이름 (get_latest_result)
    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default


# =========================================================
# 단위 테스트 실행 코드
# =========================================================
if __name__ == "__main__":
    print("[단위 테스트 시작] RealSensorSource 데이터 파싱 검증")

    try:
        sensor_source = RealSensorSource()
    except TypeError as e:
        print(f"객체 생성 실패: {e}")
        exit()

    # 테스트 데이터 입력
    dummy_payload = {
        "timestamp": "2026-02-12T19:00:00",
        "bio_arousal": 0.8,
        "bio_conf": 0.9,
        "bpm": 120.0,
        "rmssd_use": 0.03,
        "gsr_cur": 0.8,
        "gsr_slope_1s": 12.0,
        "gsr_peak_10s": 3,
        "ppgQ": 0.95,
        "finger_on": True,
        "gsr_fresh": True,
        "note": "test"
    }
    print(f"입력: {dummy_payload}")

    try:
        sensor_source.update_window(dummy_payload)

        result = sensor_source.get_latest_result()

        if result:
            print(f"결과: {result.raw_metrics}")

            hr_check = result.raw_metrics.get("hrv")
            gsr_check = result.raw_metrics.get("gsr")
            arousal_check = result.raw_metrics.get("bio_arousal")

            if hr_check == 0.03 and gsr_check == 0.8 and arousal_check == 0.8:
                print("\n결과: PASS (성공 데이터가 정확히 연결되었습니다.)")
            else:
                print(
                    f"\n결과: FAIL "
                    f"(값 불일치 -> HRV:{hr_check}, GSR:{gsr_check}, AROUSAL:{arousal_check})"
                )
        else:
            print("\n결과: FAIL (데이터가 저장되지 않음)")

    except Exception as e:
        print(f"\n에러 발생: {e}")
