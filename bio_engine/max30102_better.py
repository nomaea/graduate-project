# max30102_better.py
from smbus2 import SMBus
import time
import errno

I2C_ADDR = 0x57

REG_INTR_STATUS_1 = 0x00
REG_INTR_STATUS_2 = 0x01
REG_FIFO_WR_PTR    = 0x04
REG_OVF_COUNTER    = 0x05
REG_FIFO_RD_PTR    = 0x06
REG_FIFO_DATA      = 0x07
REG_FIFO_CONFIG    = 0x08
REG_MODE_CONFIG    = 0x09
REG_SPO2_CONFIG    = 0x0A
REG_LED1_PA        = 0x0C  # RED
REG_LED2_PA        = 0x0D  # IR

MODE_SPO2 = 0x03  # RED + IR

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

class MAX30102:
    def __init__(self, bus_id=1, address=I2C_ADDR):
        self.address = address
        self.bus = SMBus(bus_id)

    def _retry(self, fn, tries=10, delay=0.01):
        last = None
        for _ in range(tries):
            try:
                return fn()
            except BlockingIOError as e:
                if getattr(e, "errno", None) == errno.EAGAIN:
                    time.sleep(delay)
                    last = e
                    continue
                raise
            except OSError as e:
                if getattr(e, "errno", None) in (121, 110, errno.EAGAIN):
                    time.sleep(delay)
                    last = e
                    continue
                raise
        raise last if last else OSError(121, "I2C failed after retries")

    def _w8(self, reg, val):
        return self._retry(lambda: self.bus.write_byte_data(self.address, reg, val & 0xFF))

    def _r8(self, reg):
        return self._retry(lambda: self.bus.read_byte_data(self.address, reg))

    def _read_block(self, reg, n):
        return self._retry(lambda: self.bus.read_i2c_block_data(self.address, reg, n))

    def close(self):
        try:
            self.bus.close()
        except Exception:
            pass

    def reset(self):
        self._w8(REG_MODE_CONFIG, 0x40)
        time.sleep(0.05)
        try:
            _ = self._r8(REG_INTR_STATUS_1)
            _ = self._r8(REG_INTR_STATUS_2)
        except Exception:
            pass

    def fifo_flush(self):
        self._w8(REG_FIFO_WR_PTR, 0x00)
        self._w8(REG_OVF_COUNTER, 0x00)
        self._w8(REG_FIFO_RD_PTR, 0x00)

    def setup(
        self,
        mode=MODE_SPO2,
        sample_avg=4,
        sample_rate=100,
        pulse_width=411,
        adc_range=4096,
        led_red=0x30,
        led_ir=0x40
    ):
        self.reset()

        avg_map = {1:0, 2:1, 4:2, 8:3, 16:4, 32:5}
        avg_bits = avg_map.get(sample_avg, 2) << 5
        roll_bit = (1 << 4)
        a_full = 0x0F
        self._w8(REG_FIFO_CONFIG, avg_bits | roll_bit | a_full)

        adc_map = {2048:0, 4096:1, 8192:2, 16384:3}
        sr_map  = {50:0, 100:1, 200:2, 400:3, 800:4, 1000:5, 1600:6, 3200:7}
        pw_map  = {69:0, 118:1, 215:2, 411:3}

        adc_bits = adc_map.get(adc_range, 1) << 5
        sr_bits  = sr_map.get(sample_rate, 1) << 2
        pw_bits  = pw_map.get(pulse_width, 3)
        self._w8(REG_SPO2_CONFIG, adc_bits | sr_bits | pw_bits)

        self._w8(REG_LED1_PA, _clamp(led_red, 0, 255))
        self._w8(REG_LED2_PA, _clamp(led_ir, 0, 255))
        self._w8(REG_MODE_CONFIG, mode & 0x07)

        self.fifo_flush()
        time.sleep(0.02)

    def _num_samples_available(self):
        wr = self._r8(REG_FIFO_WR_PTR) & 0x1F
        rd = self._r8(REG_FIFO_RD_PTR) & 0x1F
        return (wr - rd) & 0x1F

    def read_samples(self, max_samples=32):
        try:
            n = self._num_samples_available()
        except OSError:
            try:
                self.reset()
                self.fifo_flush()
            except Exception:
                pass
            return []

        if n == 0:
            return []

        n = min(n, max_samples)
        out = []
        for _ in range(n):
            try:
                data = self._read_block(REG_FIFO_DATA, 6)
            except OSError:
                try:
                    self.reset()
                    self.fifo_flush()
                except Exception:
                    pass
                return []

            red = ((data[0] << 16) | (data[1] << 8) | data[2]) & 0x3FFFF
            ir  = ((data[3] << 16) | (data[4] << 8) | data[5]) & 0x3FFFF
            out.append((red, ir))
        return out
