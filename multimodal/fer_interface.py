from abc import ABC, abstractmethod
from typing import Optional
from .fer_types import FerResult


class FerSource(ABC):
    """
    FER(카메라) 쪽에서 결과를 가져오는 공통 인터페이스.
    실제 구현은 FER 담당 팀 / 혹은 Mock 클래스에서 상속해서 구현.
    """

    @abstractmethod
    def get_latest_result(self) -> Optional[FerResult]:
        """
        가장 최근 FER 결과를 반환.
        아직 준비된 결과가 없다면 None을 반환할 수 있다.
        """
        raise NotImplementedError
