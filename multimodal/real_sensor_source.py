from typing import Dict, Optional

# 필요한 모듈 임포트
from sensor_interface import SensorSource
from sensor_types import SensorResult
from bio_service import BioService

class RealSensorSource(SensorSource):
    """
    실제 PPG(심박) 및 GSR(피부전도도) 센서 데이터를 처리하여 
    멀티모달 엔진에 전달하는 구현체.
    """

    def __init__(self) -> None:
        self._bio = BioService()
        self._latest: Optional[SensorResult] = None

    def update_window(self, payload: Dict) -> None:
        """
        보내줄 데이터 예시:
        { "timestamp": "...", "ppg_hr": 85.5, "gsr_val": 0.25 }
        """
        # 1. BioService 실행 (에러 방지용 가짜/진짜 모델 사용)
        bio_out = self._bio.process_window(payload)
        features: Dict[str, float] = bio_out.get("features", {})

        # 2. 데이터 매핑 (수정됨: 친구가 준 값을 최우선으로!)
        # 친구가 "ppg_hr" 키로 값을 줬다면, BioService 결과(0.0)를 무시하고 친구 값을 씁니다.
        
        if "ppg_hr" in payload:
            ppg_heart_rate = float(payload["ppg_hr"])
        else:
            ppg_heart_rate = float(features.get("hr_mean", 0.0))

        if "gsr_val" in payload:
            gsr_stress_val = float(payload["gsr_val"])
        else:
            gsr_stress_val = float(features.get("eda_mean", 0.0))

        # 3. 멀티모달 엔진용 데이터 구조 생성
        raw_metrics: Dict[str, float] = {
            "hrv": ppg_heart_rate,  
            "gsr": gsr_stress_val
        }

        # 4. 점수 계산
        sensor_emotion_scores = self._calculate_sensor_scores(ppg_heart_rate, gsr_stress_val)

        # 5. 결과 저장 (timestamp 제거 버전)
        self._latest = SensorResult(
            raw_metrics=raw_metrics,
            emotion_scores=sensor_emotion_scores
        )

    def _calculate_sensor_scores(self, hr: float, gsr: float) -> Dict[str, float]:
        """간단한 스트레스 점수 계산 로직"""
        stress_score = 0.0
        if hr > 90 or gsr > 0.5:
            stress_score = min(1.0, ((hr - 60) / 100) * 0.5 + gsr * 0.5)
        
        if stress_score < 0.1: stress_score = 0.0

        return {
            "safe": 1.0 - stress_score,
            "stressed": stress_score
        }

    # 부모 클래스가 요구하는 함수 이름 (get_latest_result)
    def get_latest_result(self) -> Optional[SensorResult]:
        return self._latest

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
        "ppg_hr": 120.0,    
        "gsr_val": 0.8      
    }
    print(f"입력: {dummy_payload}")

    try:
        sensor_source.update_window(dummy_payload)
        
        result = sensor_source.get_latest_result() 

        if result:
            print(f"결과: {result.raw_metrics}")
            
            hr_check = result.raw_metrics.get("hrv")
            gsr_check = result.raw_metrics.get("gsr")

            if hr_check == 120.0 and gsr_check == 0.8:
                print("\n결과: PASS (성공 데이터가 정확히 연결되었습니다.)")
            else:
                print(f"\n결과: FAIL (값 불일치 -> HR:{hr_check}, GSR:{gsr_check})")
        else:
            print("\n결과: FAIL (데이터가 저장되지 않음)")

    except Exception as e:
        print(f"\n에러 발생: {e}")