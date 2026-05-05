import re
import time
import threading
import logging

import serial
from serial.tools import list_ports

logger = logging.getLogger(__name__)

# FIX #5: 재연결 파라미터 상수화
_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_DELAY_S = 1.0


class GSRSerialReader:
    def __init__(self, port=None, baud=9600):
        self.port = port
        self.baud = baud
        self.latest_value = None
        self.latest_ts = None
        # FIX #2: start() 실패 시 예외 대신 error 문자열 설정 — bio 스레드 크래시 방지
        self.error: str | None = None
        self._stop = False
        self._th = None
        self._ser = None

    def auto_port(self):
        candidates = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]
        devices = [p.device for p in list_ports.comports()]
        for c in candidates:
            if c in devices:
                return c
        for p in list_ports.comports():
            if "ttyUSB" in p.device or "ttyACM" in p.device:
                return p.device
        return None

    def start(self):
        if self.port is None:
            self.port = self.auto_port()

        # FIX #2: 포트 없으면 raise 대신 self.error 설정 후 return
        # 기존: raise RuntimeError → bio_engine_main 스레드 조용히 종료
        # 수정: 호출자가 gsr.error 확인 후 경고 출력, bio는 GSR 없이 계속 실행
        if self.port is None:
            self.error = "GSR serial port not found (tried ttyUSB0/ttyACM0)"
            logger.error(f"[GSR] {self.error}")
            return

        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except serial.SerialException as e:
            self.error = f"Cannot open {self.port}: {e}"
            logger.error(f"[GSR] {self.error}")
            return

        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        logger.info(f"[GSR] Started on {self.port} @ {self.baud}bps")

    def stop(self):
        self._stop = True
        if self._th:
            self._th.join(timeout=1.0)
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass

    def _parse_value(self, line: str):
        line = line.strip()
        if not line:
            return None
        m = re.search(r"(-?\d+)", line)
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    # FIX #5: 재연결 로직 추가
    def _reconnect(self) -> bool:
        logger.warning(f"[GSR] Serial disconnected — attempting reconnect to {self.port}")
        for attempt in range(1, _MAX_RECONNECT_ATTEMPTS + 1):
            if self._stop:
                return False
            try:
                if self._ser:
                    self._ser.close()
            except Exception:
                pass
            time.sleep(_RECONNECT_DELAY_S)
            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
                logger.info(f"[GSR] Reconnected on attempt {attempt}")
                return True
            except serial.SerialException as e:
                logger.warning(f"[GSR] Reconnect attempt {attempt}/{_MAX_RECONNECT_ATTEMPTS} failed: {e}")
        logger.error(f"[GSR] Could not reconnect after {_MAX_RECONNECT_ATTEMPTS} attempts — stopping reader")
        return False

    # FIX #5: 예외 종류별 처리 + 오류 로그 추가
    # 기존: except Exception: time.sleep(0.05) — 모든 오류 묵살, 재연결 없음
    def _loop(self):
        while not self._stop:
            try:
                raw = self._ser.readline().decode(errors="ignore")
                v = self._parse_value(raw)
                if v is not None:
                    self.latest_value = v
                    self.latest_ts = time.time()
            except serial.SerialException as e:
                # 포트 연결 해제 → 재연결 시도
                logger.warning(f"[GSR] SerialException: {e}")
                if not self._reconnect():
                    self._stop = True  # 재연결 실패 시 루프 종료
            except Exception as e:
                # 기타 예외는 로그 남기고 짧게 대기 후 재시도
                logger.warning(f"[GSR] Unexpected error: {type(e).__name__}: {e}")
                time.sleep(0.05)
