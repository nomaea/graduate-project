# bio_engine_v1.py
import time
import csv
import re
import threading
from collections import deque

import serial
from serial.tools import list_ports

from max30102_better import MAX30102

# ----------------------------
# Serial (GSR) reader thread
# ----------------------------
class GSRSerialReader:
    """
    Reads GSR values from Arduino via USB serial.
    Accepts formats like:
      "GSR: 512"
      "512"
      "gsr=512"
    """
    def __init__(self, port=None, baud=9600):
        self.port = port
        self.baud = baud
        self.latest_value = None
        self.latest_ts = None
        self._stop = False
        self._th = None
        self._ser = None

    def auto_port(self):
        # try common first
        candidates = ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyUSB1", "/dev/ttyACM1"]
        for c in candidates:
            try:
                return c if c in [p.device for p in list_ports.comports()] else None
            except Exception:
                pass

        # fallback: pick first usb/acm
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
# Helpers
# ----------------------------
def clamp01(x):
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def norm_range(x, lo, hi):
    if hi <= lo:
        return 0.0
    return clamp01((x - lo) / (hi - lo))

def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ----------------------------
# Main BIO Engine v1
# ----------------------------
def main():
    # ---- config ----
    FS = 1.0  # output rate (Hz)
    PRINT_EVERY = 1.0

    # Finger-on threshold (based on your logs)
    IR_FINGER_MIN = 50000

    # Heuristic normalization ranges (tune later)
    # bpm: resting 60~100, but can go higher
    BPM_LO, BPM_HI = 55, 120

    # gsr raw is arbitrary; we normalize by short-term dynamics instead:
    # slope in raw units per second
    GSR_SLOPE_LO, GSR_SLOPE_HI = 0, 50

    # ---- init sensors ----
    max3 = MAX30102()
    max3.setup(
        sample_avg=4,
        sample_rate=100,
        pulse_width=411,
        adc_range=4096,
        led_red=0x30,
        led_ir=0x40
    )

    gsr = GSRSerialReader(port=None, baud=9600)
    gsr.start()
    print(f"[INFO] GSR serial port = {gsr.port}")

    # ---- buffers (timestamped) ----
    # store tuples (t, value)
    bpm_buf = deque()      # last 10s
    ir_buf  = deque()      # last 5s
    gsr_buf = deque()      # last 10s

    # for peak counting (10s)
    gsr_slope_buf = deque()  # last 10s slopes

    # last known bpm estimate (simple, from IR peaks is heavier; start with placeholder)
    # If you already have your BPM code, you can plug it in here.
    # For v1, we'll estimate "bpm_proxy" using IR variance (not medical) unless you provide BPM.
    bpm_est = None

    # ---- CSV ----
    csv_name = f"bio_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    f = open(csv_name, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts_iso", "ts_unix",
        "red", "ir",
        "bpm_est",
        "gsr",
        "bpm_mean_5s", "bpm_std_10s",
        "ir_mean_1s", "ir_std_5s",
        "gsr_mean_5s", "gsr_slope_1s", "gsr_peak_10s",
        "bio_arousal_score", "bio_confidence",
        "finger_on", "gsr_fresh"
    ])
    f.flush()
    print(f"[INFO] Logging to {csv_name}")

    last_print = time.time()
    next_tick = time.monotonic()

    try:
        while True:
            # ---- sample MAX30102 (take latest sample from FIFO) ----
            samples = max3.read_samples(max_samples=32)
            red, ir = (None, None)
            if samples:
                red, ir = samples[-1]

            ts = time.time()

            # maintain IR buffer
            if ir is not None:
                ir_buf.append((ts, ir))
                # keep 5s
                while ir_buf and (ts - ir_buf[0][0]) > 5.0:
                    ir_buf.popleft()

            finger_on = (ir is not None and ir >= IR_FINGER_MIN)

            # ---- read latest GSR ----
            gsr_val = gsr.latest_value
            gsr_fresh = (gsr.latest_ts is not None and (ts - gsr.latest_ts) <= 1.0)

            if gsr_val is not None:
                gsr_buf.append((ts, gsr_val))
                while gsr_buf and (ts - gsr_buf[0][0]) > 10.0:
                    gsr_buf.popleft()

            # ---- v1 BPM estimate ----
            # If you have a dedicated BPM estimator, replace this block with it.
            # Here we do a very rough proxy:
            # - if finger is on and IR buffer exists, keep previous bpm_est or set None
            # You can also manually input bpm_est from your working bpm script.
            if not finger_on:
                bpm_est = None
                bpm_buf.clear()
            else:
                # keep bpm_est if already computed elsewhere; otherwise remain None
                pass

            if bpm_est is not None:
                bpm_buf.append((ts, float(bpm_est)))
                while bpm_buf and (ts - bpm_buf[0][0]) > 10.0:
                    bpm_buf.popleft()

            # ---- feature calc helpers ----
            def mean_std(buf, window_s):
                vals = [v for (t, v) in buf if (ts - t) <= window_s]
                if not vals:
                    return None, None
                m = sum(vals) / len(vals)
                if len(vals) < 2:
                    return m, 0.0
                var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
                return m, var ** 0.5

            # bpm features
            bpm_mean_5s, _ = mean_std(bpm_buf, 5.0)
            _, bpm_std_10s = mean_std(bpm_buf, 10.0)

            # IR features
            ir_mean_1s, _ = mean_std(ir_buf, 1.0)
            _, ir_std_5s = mean_std(ir_buf, 5.0)

            # GSR features
            gsr_mean_5s, _ = mean_std(gsr_buf, 5.0)

            # gsr slope 1s: current - value 1s ago
            gsr_slope_1s = None
            if gsr_buf:
                cur = gsr_buf[-1][1]
                past_candidates = [v for (t, v) in gsr_buf if (ts - t) >= 1.0]
                if past_candidates:
                    # take the oldest that is >= 1s ago (approx)
                    past = past_candidates[0]
                    gsr_slope_1s = float(cur - past)
                else:
                    gsr_slope_1s = 0.0

            # peak count in last 10s based on slope threshold
            if gsr_slope_1s is not None:
                gsr_slope_buf.append((ts, gsr_slope_1s))
                while gsr_slope_buf and (ts - gsr_slope_buf[0][0]) > 10.0:
                    gsr_slope_buf.popleft()

            # count "peaks" as slopes above threshold
            peak_thr = 20.0
            gsr_peak_10s = sum(1 for (_, s) in gsr_slope_buf if s is not None and s > peak_thr)

            # ---- bio_confidence ----
            conf = 1.0
            if not finger_on:
                conf *= 0.0
            if not gsr_fresh:
                conf *= 0.3  # degrade but not zero
            # if IR is weirdly low/high sudden drop, reduce confidence
            if ir is None:
                conf *= 0.0
            else:
                # if IR too low, already finger_off
                pass

            bio_conf = clamp01(conf)

            # ---- arousal score (heuristic v1) ----
            # If bpm not available, we rely more on GSR dynamics.
            bpm_component = None
            if bpm_mean_5s is not None:
                bpm_component = norm_range(bpm_mean_5s, BPM_LO, BPM_HI)

            gsr_component = 0.0
            if gsr_slope_1s is not None:
                gsr_component = norm_range(abs(gsr_slope_1s), GSR_SLOPE_LO, GSR_SLOPE_HI)

            peak_component = norm_range(gsr_peak_10s, 0, 10)

            # weights (v1)
            if bpm_component is None:
                arousal = 0.65 * gsr_component + 0.35 * peak_component
            else:
                arousal = 0.40 * bpm_component + 0.40 * gsr_component + 0.20 * peak_component

            arousal = clamp01(arousal)

            # final outputs should respect confidence
            arousal_out = arousal * bio_conf

            # ---- log ----
            w.writerow([
                now_iso(), f"{ts:.3f}",
                red if red is not None else "",
                ir if ir is not None else "",
                f"{bpm_est:.2f}" if bpm_est is not None else "",
                gsr_val if gsr_val is not None else "",
                f"{bpm_mean_5s:.2f}" if bpm_mean_5s is not None else "",
                f"{bpm_std_10s:.2f}" if bpm_std_10s is not None else "",
                f"{ir_mean_1s:.2f}" if ir_mean_1s is not None else "",
                f"{ir_std_5s:.2f}" if ir_std_5s is not None else "",
                f"{gsr_mean_5s:.2f}" if gsr_mean_5s is not None else "",
                f"{gsr_slope_1s:.2f}" if gsr_slope_1s is not None else "",
                int(gsr_peak_10s),
                f"{arousal_out:.3f}",
                f"{bio_conf:.3f}",
                int(finger_on),
                int(gsr_fresh),
            ])
            f.flush()

            # ---- print ----
            if (time.time() - last_print) >= PRINT_EVERY:
                last_print = time.time()
                print(
                    f"IR={ir if ir is not None else 'NA'} "
                    f"GSR={gsr_val if gsr_val is not None else 'NA'} "
                    f"arousal={arousal_out:.2f} conf={bio_conf:.2f} "
                    f"finger_on={int(finger_on)} gsr_fresh={int(gsr_fresh)}"
                )

            # ---- 1Hz tick ----
            next_tick += 1.0 / FS
            sleep_for = next_tick - time.monotonic()
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
