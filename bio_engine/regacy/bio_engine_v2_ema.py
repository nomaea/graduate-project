# bio_engine_v2.py
import os
import time
import csv
import re
import math
import threading
from collections import deque

import serial
from serial.tools import list_ports

from max30102_better import MAX30102


# ----------------------------
# Utils
# ----------------------------
def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def norm_range(x, lo, hi) -> float:
    if x is None:
        return 0.0
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))

def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def stdev(xs):
    if xs is None or len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5

def median(xs):
    if not xs:
        return None
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])

def mad(xs):
    if not xs:
        return None
    m = median(xs)
    dev = [abs(x - m) for x in xs]
    return median(dev)


# ----------------------------
# GSR Serial Reader
# ----------------------------
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


# ----------------------------
# PPG Processor (BPM + RR + HRV)
# ----------------------------
class PPGProcessor:
    def __init__(self, fs_hz: int, finger_ir_min: int = 50000, bpm_tau_s: float = 1.5):
        self.fs = fs_hz
        self.finger_ir_min = finger_ir_min

        self.ir = deque()        # (t, ir)
        self.ir_filt = deque()   # (t, y)
        self.peaks = deque()     # peak times
        self.rr = deque()        # (t, rr)

        self.last_peak_t = None
        self._ma = deque(maxlen=5)

        # EMA-smoothed BPM (for low-latency display)
        self.bpm_tau = float(bpm_tau_s) if bpm_tau_s and bpm_tau_s > 0 else 1.5
        self.bpm_ema = None

    def _trim(self, now_t: float):
        while self.ir and (now_t - self.ir[0][0]) > 15.0:
            self.ir.popleft()
        while self.ir_filt and (now_t - self.ir_filt[0][0]) > 15.0:
            self.ir_filt.popleft()
        while self.peaks and (now_t - self.peaks[0]) > 60.0:
            self.peaks.popleft()
        while self.rr and (now_t - self.rr[0][0]) > 60.0:
            self.rr.popleft()

    def finger_on(self) -> bool:
        return bool(self.ir) and (self.ir[-1][1] >= self.finger_ir_min)

    def _reset_keep_raw(self):
        self.ir_filt.clear()
        self.peaks.clear()
        self.rr.clear()
        self.last_peak_t = None
        self._ma.clear()
        self.bpm_ema = None

    def add_samples(self, samples, now_t: float):
        if not samples:
            return

        n = len(samples)
        dt = 1.0 / float(self.fs)
        start_t = now_t - (n - 1) * dt

        for i, (_, ir) in enumerate(samples):
            t = start_t + i * dt
            self.ir.append((t, ir))

            if ir < self.finger_ir_min:
                self._reset_keep_raw()
                continue

            # detrend by median of last 2s raw
            window_raw = [v for (tt, v) in self.ir if (t - tt) <= 2.0]
            med = median(window_raw) if window_raw else ir
            y = ir - (med if med is not None else ir)

            # smooth
            self._ma.append(y)
            y_sm = sum(self._ma) / len(self._ma)

            self.ir_filt.append((t, y_sm))
            self._detect_peak(t)

        self._trim(now_t)

    # ---- tuned peak detector (for IR ~ 250k environment) ----
    def _detect_peak(self, t: float):
        if len(self.ir_filt) < 5:
            return

        (t1, y1) = self.ir_filt[-3]
        (t2, y2) = self.ir_filt[-2]
        (t3, y3) = self.ir_filt[-1]

        # local max at middle
        if not (y2 > y1 and y2 > y3):
            return

        # refractory (tuned): 0.30 -> 0.25 to reduce missed peaks
        if self.last_peak_t is not None and (t2 - self.last_peak_t) < 0.25:
            return

        recent = [y for (tt, y) in self.ir_filt if (t2 - tt) <= 5.0]
        if len(recent) < 20:
            return

        m = median(recent)
        d = mad(recent)

        # tuned: MAD threshold 5.0 -> 2.0 (allow weaker pulses)
        if d is None or d < 2.0:
            return

        # tuned: threshold factor 1.0 -> 0.6 (more sensitive)
        thr = (m if m is not None else 0.0) + 0.6 * d

        if y2 < thr:
            return

        # accept peak
        self.peaks.append(t2)
        if self.last_peak_t is not None:
            rr = t2 - self.last_peak_t
            # tuned: RR lower bound 0.30 -> 0.28 (help early RR accumulation)
            if 0.28 <= rr <= 2.00:
                self.rr.append((t2, rr))

                # Update EMA BPM on every valid RR (fast response, less jitter than raw)
                inst_bpm = 60.0 / rr
                tau = self.bpm_tau if self.bpm_tau > 0 else 1.5
                alpha = 1.0 - math.exp(-rr / tau)  # time-constant based smoothing
                if self.bpm_ema is None:
                    self.bpm_ema = inst_bpm
                else:
                    self.bpm_ema += alpha * (inst_bpm - self.bpm_ema)
        self.last_peak_t = t2

    def bpm(self):
        """EMA-smoothed BPM for low-latency display."""
        if not self.ir:
            return None
        if not self.rr:
            return None
        if self.bpm_ema is not None:
            return self.bpm_ema

        # Fallback: average over last 8s (should rarely happen)
        now_t = self.ir[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 8.0]
        if len(rrs) < 2:
            return None
        avg_rr = sum(rrs) / len(rrs)
        return None if avg_rr <= 0 else (60.0 / avg_rr)

    def rmssd_30s(self):
        if not self.ir:
            return None
        now_t = self.ir[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 30.0]
        if len(rrs) < 3:
            return None
        diffs = [(rrs[i] - rrs[i-1]) for i in range(1, len(rrs))]
        sq = [d*d for d in diffs]
        return (sum(sq) / len(sq)) ** 0.5

    def signal_quality(self):
        if not self.finger_on():
            return 0.0
        if not self.ir:
            return 0.0
        now_t = self.ir[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 10.0]
        if len(rrs) < 3:
            return 0.2
        rr_std = stdev(rrs)
        q = 1.0 - norm_range(rr_std, 0.02, 0.20)
        if len(rrs) < 4:
            q *= 0.6
        return clamp01(q)


# ----------------------------
# BIO Engine v2 main
# ----------------------------
def main():
    # ---- config ----
    PPG_FS = 100
    READ_HZ = 50.0  # PPG read loop Hz (>=20 recommended to avoid FIFO overflow)
    OUT_HZ = 1.0    # print/CSV Hz
    BPM_EMA_TAU = 1.5  # seconds: lower=faster response, higher=smoother
    IR_FINGER_MIN = 50000

    # normalization ranges (tune later)
    BPM_LO, BPM_HI = 55, 120
    RMSSD_LO, RMSSD_HI = 0.01, 0.12
    GSR_SLOPE_LO, GSR_SLOPE_HI = 0, 20   # tuned: 60 -> 20

    # gsr peak threshold tuned
    GSR_PEAK_THR = 10.0                  # tuned: 20 -> 10

    # ---- log dir ----
    base = os.path.join(os.getcwd(), "logs", "bio", time.strftime("%Y%m%d"))
    os.makedirs(base, exist_ok=True)
    csv_path = os.path.join(base, f"bio_v2_{time.strftime('%H%M%S')}.csv")

    # ---- init sensors ----
    max3 = MAX30102()
    max3.setup(
        sample_avg=4,
        sample_rate=PPG_FS,
        pulse_width=411,
        adc_range=4096,
        led_red=0x30,
        led_ir=0x40
    )

    gsr = GSRSerialReader(port=None, baud=9600)
    gsr.start()
    print(f"[INFO] GSR serial port = {gsr.port}")

    ppg = PPGProcessor(fs_hz=PPG_FS, finger_ir_min=IR_FINGER_MIN, bpm_tau_s=BPM_EMA_TAU)

    # ---- buffers for GSR features ----
    gsr_buf = deque()        # (t, gsr)
    gsr_slope_buf = deque()  # (t, slope)

    def trim_buf(buf, now_t, secs):
        while buf and (now_t - buf[0][0]) > secs:
            buf.popleft()

    # ---- CSV ----
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts_iso","ts_unix","red","ir",
        "bpm","rmssd_s",
        "gsr","gsr_mean_5s","gsr_slope_1s","gsr_peak_10s",
        "bpm_c","rmssd_inv","gsr_slope_c","gsr_peak_c",
        "bio_arousal","bio_conf","ppgQ","finger_on","gsr_fresh","note"
    ])
    f.flush()
    print(f"[INFO] Logging to {csv_path}")

    next_read = time.monotonic()
    next_out = time.monotonic()
    last_gsr_ts = None  # last timestamp seen from GSR thread
    last_red, last_ir = (None, None)

    try:
        while True:
            now_t = time.time()

            # ---- fast read MAX30102 (must be frequent to avoid FIFO overflow) ----
            samples = max3.read_samples(max_samples=32)
            if samples:
                last_red, last_ir = samples[-1]
                ppg.add_samples(samples, now_t=now_t)

            # ---- update GSR buffer only when a new reading arrives ----
            gsr_val = gsr.latest_value
            gsr_ts = gsr.latest_ts
            gsr_fresh = (gsr_ts is not None and (now_t - gsr_ts) <= 1.0)

            if gsr_val is not None and gsr_ts is not None and gsr_ts != last_gsr_ts:
                last_gsr_ts = gsr_ts
                gsr_buf.append((gsr_ts, int(gsr_val)))
                trim_buf(gsr_buf, now_t, 10.0)

            # ---- output/log tick @ OUT_HZ ----
            if time.monotonic() >= next_out:
                next_out += 1.0 / OUT_HZ

                finger_on = ppg.finger_on()

                gsr_5s = [v for (t, v) in gsr_buf if (now_t - t) <= 5.0]
                gsr_mean_5s = mean(gsr_5s)

                # slope 1s (nearest sample <= 1s ago)
                gsr_slope_1s = None
                if gsr_buf:
                    cur = gsr_buf[-1][1]
                    target_t = now_t - 1.0
                    past = None
                    for (t, v) in reversed(gsr_buf):
                        if t <= target_t:
                            past = v
                            break
                    gsr_slope_1s = float(cur - past) if past is not None else 0.0

                if gsr_slope_1s is not None:
                    gsr_slope_buf.append((now_t, gsr_slope_1s))
                    trim_buf(gsr_slope_buf, now_t, 10.0)

                gsr_peak_10s = sum(1 for (_, s) in gsr_slope_buf if s is not None and s > GSR_PEAK_THR)

                # ---- PPG features ----
                bpm = ppg.bpm()
                rmssd = ppg.rmssd_30s()
                ppg_q = ppg.signal_quality()

                # ---- bio_confidence ----
                conf = 1.0
                if not finger_on:
                    conf *= 0.0
                if not gsr_fresh:
                    conf *= 0.4
                conf *= (0.3 + 0.7 * ppg_q)
                bio_conf = clamp01(conf)

                # ---- normalized components ----
                bpm_c = norm_range(bpm, BPM_LO, BPM_HI) if bpm is not None else None

                rmssd_inv = None
                if rmssd is not None:
                    rmssd_inv = 1.0 - norm_range(rmssd, RMSSD_LO, RMSSD_HI)

                gsr_slope_c = norm_range(abs(gsr_slope_1s) if gsr_slope_1s is not None else None,
                                         GSR_SLOPE_LO, GSR_SLOPE_HI)
                gsr_peak_c = norm_range(gsr_peak_10s, 0, 10)

                # ---- arousal_raw ----
                if rmssd_inv is None and bpm_c is None:
                    arousal = 0.70 * gsr_slope_c + 0.30 * gsr_peak_c
                elif rmssd_inv is None:
                    arousal = 0.45 * bpm_c + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                else:
                    if bpm_c is None:
                        arousal = 0.45 * rmssd_inv + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                    else:
                        arousal = 0.30 * bpm_c + 0.40 * rmssd_inv + 0.20 * gsr_slope_c + 0.10 * gsr_peak_c

                arousal = clamp01(arousal)

                # tuned: do NOT multiply by confidence (keep separate)
                bio_arousal = arousal

                note = ""
                if not finger_on:
                    note = "finger_off"
                elif rmssd is None:
                    note = "hrv_warming_up(need~30s)"

                # ---- CSV (rounded) ----
                w.writerow([
                    now_iso(), f"{now_t:.3f}",
                    last_red if last_red is not None else "",
                    last_ir if last_ir is not None else "",
                    f"{bpm:.1f}" if bpm is not None else "",
                    f"{rmssd:.3f}" if rmssd is not None else "",
                    gsr_val if gsr_val is not None else "",
                    f"{gsr_mean_5s:.1f}" if gsr_mean_5s is not None else "",
                    f"{gsr_slope_1s:.1f}" if gsr_slope_1s is not None else "",
                    int(gsr_peak_10s),
                    f"{bpm_c:.2f}" if bpm_c is not None else "",
                    f"{rmssd_inv:.2f}" if rmssd_inv is not None else "",
                    f"{gsr_slope_c:.2f}",
                    f"{gsr_peak_c:.2f}",
                    f"{bio_arousal:.1f}",
                    f"{bio_conf:.1f}",
                    f"{ppg_q:.1f}",
                    int(finger_on),
                    int(gsr_fresh),
                    note
                ])
                f.flush()

                # ---- print (1 decimal) ----
                bpm_s = f"{bpm:.1f}" if bpm is not None else "..."
                rmssd_s = f"{rmssd:.3f}" if rmssd is not None else "..."
                gsr_s = f"{gsr_val}" if gsr_val is not None else "..."
                print(
                    f"BPM={bpm_s} RMSSD={rmssd_s} GSR={gsr_s} "
                    f"arousal={bio_arousal:.1f} conf={bio_conf:.1f} "
                    f"ppgQ={ppg_q:.1f} finger_on={int(finger_on)} note={note}"
                )

            # ---- read tick @ READ_HZ ----
            next_read += 1.0 / READ_HZ
            sleep_for = next_read - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\n[INFO] Exit")
    finally:
        try:
            gsr.stop()
        except Exception:
            pass
        try:
            max3.close()
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
