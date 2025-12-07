from typing import Dict


class BioRuleModel:
    """
    특징 추출된 생체신호 값을 기반으로 스트레스/안정 상태를 계산하는 규칙 기반 모델
    심박수 피부전도·가속도 특징값을 기준으로 스트레스 점수를 산출하고 안정/스트레스 확률로 변환
    향후 머신러닝 모델로 교체될 수 있도록 독립된 판단 모듈로 구성

    """

    def __init__(
        self,
        hr_threshold: float = 90.0,
        eda_threshold: float = 0.15,
        acc_high_threshold: float = 0.15,
        acc_low_threshold: float = 0.02,
    ):
        self.hr_threshold = hr_threshold
        self.eda_threshold = eda_threshold
        self.acc_high_threshold = acc_high_threshold
        self.acc_low_threshold = acc_low_threshold

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        hr = features.get("hr_mean", 0.0)
        eda = features.get("eda_mean", 0.0)
        acc = features.get("acc_mean", 0.0)

        stress_score = 0.0

        # 모든 스코어링 규칙은 지속적으로 세부 조정 필요
        # 기본 스트레스 규칙
        if hr > self.hr_threshold:
            stress_score += 0.5
        if eda > self.eda_threshold:
            stress_score += 0.5

        # 가속도계 데이터를 활용하여 사용자가 활동 중임을 감지
        # 움직임이 큰 경우 운동 노이즈일 가능성 → 스트레스 점수 감소 
        if acc > self.acc_high_threshold:
            stress_score *= 0.7

        # 움직임이 거의 없는데 스트레스 신호가 있으면 점수 소폭 증가
        if acc < self.acc_low_threshold and stress_score > 0:
            stress_score = min(1.0, stress_score + 0.1)

        stress_score = max(0.0, min(1.0, stress_score))
        safe_score = 1.0 - stress_score

        return {
            "bio_safe": safe_score,
            "bio_stress": stress_score,
        }
