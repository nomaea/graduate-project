# real_fer_source.py
from typing import Optional, Dict
from queue import Queue

from fer_interface import FerSource
from fer_types import FerResult

# FER 친구 레이블 순서 (fer_core.py 기준)
LABEL_ORDER = ["neutral", "happy", "sad", "angry"]


class RealFerSource(FerSource):
    """
    FER 친구가 fer_queue에 넣어준 FERPacket(딕셔너리)을 꺼내서
    멀티모달 엔진이 쓰는 FerResult로 변환.
    """

    def __init__(self, fer_queue: Queue) -> None:
        # FER 친구 큐를 직접 받아서 씀
        self._fer_queue = fer_queue
        self._latest: Optional[FerResult] = None

    def get_latest_result(self) -> Optional[FerResult]:
        if not self._fer_queue.empty():
            packet = self._fer_queue.get()
            self._latest = self._convert(packet)
        return self._latest

    def _convert(self, packet: dict) -> FerResult:
        """
        FERPacket 딕셔너리 → FerResult 변환

        FER 친구 포맷:
        {
            "softmax": [neutral, happy, sad, angry],
            "argmax_label": "neutral",
            "face_detected": True,
            ...
        }
        """
        softmax = packet.get("softmax", [0.0, 0.0, 0.0, 0.0])

        # softmax 순서 ["neutral", "happy", "sad", "angry"] 에 맞게 매핑
        emotion_scores: Dict[str, float] = {
            label: float(softmax[i])
            for i, label in enumerate(LABEL_ORDER)
        }

        # drowsy는 FER 친구가 따로 안 보내주므로
        # face_detected 값으로 alert/drowsy 추정
        face_detected = packet.get("face_detected", True)
        drowsy_scores: Dict[str, float] = {
            "alert":  1.0 if face_detected else 0.0,
            "drowsy": 0.0 if face_detected else 1.0,
        }

        return FerResult(
            emotion_scores=emotion_scores,
            drowsy_scores=drowsy_scores,
        )


# =========================================================
# 단위 테스트
# =========================================================
if __name__ == "__main__":
    print("[단위 테스트 시작] RealFerSource FER 친구 포맷 변환 검증")

    from queue import Queue

    # 1. FER 친구가 넣어줄 가짜 FERPacket
    fer_queue = Queue(maxsize=1)
    dummy_packet = {
        "ts": 12345.678,
        "seq": 1,
        "class_order": ["neutral", "happy", "sad", "angry"],
        "softmax": [0.10, 0.05, 0.03, 0.82],  # angry가 제일 높음
        "argmax_idx": 3,
        "argmax_label": "angry",
        "source": "fer",
        "face_detected": True,
        "latency_ms": 38.4
    }
    fer_queue.put(dummy_packet)
    print(f"[주입] FERPacket: {dummy_packet['argmax_label']} softmax={dummy_packet['softmax']}")

    # 2. 변환 검증
    fer_source = RealFerSource(fer_queue=fer_queue)
    result = fer_source.get_latest_result()

    if result:
        print(f"\n[변환 결과]")
        print(f"  emotion_scores: {result.emotion_scores}")
        print(f"  drowsy_scores : {result.drowsy_scores}")

        if round(result.emotion_scores.get("angry", 0), 2) == 0.82:
            print("\n결과: PASS (FER 친구 포맷 정상 변환!)")
        else:
            print("\n결과: FAIL (값 불일치)")
    else:
        print("\n결과: FAIL (데이터 없음)")