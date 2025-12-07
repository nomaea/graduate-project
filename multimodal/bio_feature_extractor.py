from typing import Dict, List
import numpy as np

"""
BIO 입력 JSON에서 심박수, 피부전도, 가속도 신호 시퀀스를 읽어 통계 기반 특징값을 추출하는 모듈
모든 생체신호는 평균 표준편차 등 기본적인 특성으로 변환되어 BIO 모델에 전달
BIO 분석 파이프라인의 첫 단계로, 입력 데이터를 모델이 처리 가능한 형태로 만드는 역할
"""

def _mean(data: List[float]) -> float:
    arr = np.asarray(data, dtype=float)
    return float(arr.mean()) if arr.size > 0 else 0.0


def _std(data: List[float]) -> float:
    arr = np.asarray(data, dtype=float)
    return float(arr.std()) if arr.size > 0 else 0.0


def extract_features(payload: Dict) -> Dict:
    """
    hr_series(심박수), eda_series(피부전도), acc_series(가속도)에서 기본 통계 특징을 추출한다.
    """
    hr_values = payload.get("hr_series") or []
    eda_values = payload.get("eda_series") or []
    acc_values = payload.get("acc_series") or []

    features = {
        "hr_mean": _mean(hr_values),
        "hr_std": _std(hr_values),
        "eda_mean": _mean(eda_values),
        "eda_std": _std(eda_values),
        "acc_mean": _mean(acc_values),
        "acc_std": _std(acc_values),
    }

    return features
