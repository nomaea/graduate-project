from typing import Dict

from bio_feature_extractor import extract_features
from bio_model import BioRuleModel


class BioService:
    """
    BIO 엔진의 전체 흐름을 담당하며 입력 JSON을 받아 특징 추출과 모델 예측을 수행
    예측된 결과를 JSON 형태로 반환하여 다른 모듈로 전달
    테스트 및 외부 호출 진입점 역할을 하며 BIO 엔진의 핵심 인터페이스 제공

    """

    def __init__(self):
        self.model = BioRuleModel()

    def process_window(self, payload: Dict) -> Dict:
        features = extract_features(payload)
        probs = self.model.predict_proba(features)

        result = {
            "timestamp": payload.get("timestamp"),
            "user_id": payload.get("user_id"),
            "features": features,
            "bio_safe": probs["bio_safe"],
            "bio_stress": probs["bio_stress"],
        }

        return result


def print_pretty(result: Dict):
    
    print("\n================= BIO 감정 분석 결과 =================")
    print(f"타임스탬프       : {result.get('timestamp')}")
    print("------------------------------------------------------")

    f = result["features"]
    print("생체신호 특징값")
    print(f" - 심박수 평균        : {f['hr_mean']:.3f}")
    print(f" - 심박수 표준편차    : {f['hr_std']:.3f}")
    print(f" - 피부전도 평균      : {f['eda_mean']:.3f}")
    print(f" - 피부전도 표준편차  : {f['eda_std']:.3f}")
    print(f" - 가속도 평균        : {f['acc_mean']:.3f}")
    print(f" - 가속도 표준편차    : {f['acc_std']:.3f}")

    print("------------------------------------------------------")
    print("감정 추론 결과")
    print(f" - 안정 상태(safe)    : {result['bio_safe']:.3f}")
    print(f" - 스트레스(stress)   : {result['bio_stress']:.3f}")
    print("======================================================\n")


if __name__ == "__main__":

    #임시로 막 넣은 샘플 데이터
    sample = {
        "timestamp": "2025-11-17T12:00:00Z",
        "user_id": "user-001",
        "hr_series": [88, 92, 95, 97, 93],
        "eda_series": [0.18, 0.21, 0.20, 0.19],
        "acc_series": [0.03, 0.05, 0.04],
    }

    service = BioService()
    result = service.process_window(sample)
    print_pretty(result)