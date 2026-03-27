import re
import time
import threading

import serial
from serial.tools import list_ports


class GSRSerialReader:
    def __init__(self, port=None, baud=9600):
        self.port = port
        self.baud = baud
        self.latest_value = None
        self.latest_ts = None
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
        if self.port is None:
            raise RuntimeError("GSR serial port not found. Try /dev/ttyUSB0 or /dev/ttyACM0")

        self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

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

    def _loop(self):
        while not self._stop:
            try:
                raw = self._ser.readline().decode(errors="ignore")
                v = self._parse_value(raw)
                if v is not None:
                    self.latest_value = v
                    self.latest_ts = time.time()
            except Exception:
                time.sleep(0.05)
