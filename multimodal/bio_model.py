# 파일명: bio_model.py

class BioRuleModel:
    """
    테스트를 위한 가짜(Dummy) 모델.
    BioService가 좋아하는 '딕셔너리' 형태로 결과를 줍니다.
    """
    def __init__(self):
        pass

    def predict(self, input_data):
        return 0.0

    def predict_proba(self, input_data):
        # BioService가 "bio_safe"라는 키를 찾고 있어서 이렇게 줘야 합니다.
        return {
            "bio_safe": 0.5,
            "bio_stress": 0.5
        }