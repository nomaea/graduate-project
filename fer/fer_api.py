from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from queue import Queue, Empty, Full


@dataclass
class FERPacket:
    ts: float
    seq: int
    class_order: List[str]
    softmax: List[float]
    argmax_idx: int
    argmax_label: str
    source: str = "fer"
    face_detected: Optional[bool] = None
    latency_ms: Optional[float] = None
    ear: Optional[float] = None        # 양안 평균 Eye Aspect Ratio
    is_drowsy: Optional[bool] = None   # EAR 기반 졸음 여부 (2초 이상 눈 감음)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FERQueuePublisher:
    """
    최신값 우선 queue publisher

    - queue maxsize=1 사용 권장
    - queue가 가득 차 있으면 기존 항목을 버리고 최신 항목으로 교체
    """

    def __init__(self, out_queue: Queue, debug_print: bool = False):
        self.out_queue = out_queue
        self.debug_print = debug_print

    def publish(self, packet: FERPacket):
        payload = packet.to_dict()

        try:
            self.out_queue.put_nowait(payload)
        except Full:
            try:
                self.out_queue.get_nowait()   # 오래된 값 제거
            except Empty:
                pass
            self.out_queue.put_nowait(payload)

        if self.debug_print:
            print(
                f"[FER->QUEUE] seq={payload['seq']} "
                f"ts={payload['ts']:.3f} "
                f"label={payload['argmax_label']} "
                f"softmax={payload['softmax']}"
            )