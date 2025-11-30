# sensor_interface.py
from abc import ABC, abstractmethod
from typing import Optional
from sensor_types import SensorResult


class SensorSource(ABC):
    """
    생체신호 결과를 제공하는 모든 소스의 공통 인터페이스.
    """

    @abstractmethod
    def get_latest_result(self) -> Optional[SensorResult]:
        """
        최신 Sensor 결과를 반환. 아직 준비 안 되었으면 None 반환.
        """
        raise NotImplementedError
