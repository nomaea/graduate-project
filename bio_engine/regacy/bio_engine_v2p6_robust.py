# bio_engine_v2p6_robust.py
#
# 목적: v2p5(즉시 출력) 유지 + "노이즈로 인한 튐"을 더 강하게 억제
# - RR 기반 BPM(EMA) + RR 아티팩트 제거(최근 중앙값 대비 급점프 배제)
# - RMSSD: 최소 RR 개수 요구 + EMA 스무딩
# - 추가(핵심): 출력 레벨의 robust(중앙값/MAD) outlier 억제 + 품질 낮으면 hold
# - 추가(핵심): GSR 원시값에 median(3) 필터 후 slope/peak 계산 (시리얼 글리치 억제)
# - 추가(선택): arousal에 "fast attack / slow release" EMA 적용(튀는 값 완화)
#
# 설치:
#   python3 -m pip install smbus2 pyserial
# 실행:
#   python3 bio_engine_v2p6_robust.py
#
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

def robust_threshold_from_mad(xs, k=3.0, abs_floor=0.0):
    """
    MAD 기반 robust threshold.
      thr = max(abs_floor, k * 1.4826 * MAD)
    """
    d = mad(xs)
    if d is None:
        return abs_floor
    thr = k * 1.4826 * d
    return max(abs_floor, thr)


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
    """PPG peak -> RR -> BPM(EMA) + RMSSD.

    - RR 아티팩트 제거: 최근 RR 중앙값 대비 일정 비율 이상 튀면 reject
    """
    def __init__(
        self,
        fs_hz: int,
        finger_ir_min: int = 50000,
        bpm_tau_s: float = 1.5,
        rr_jump_reject: float = 0.20,
    ):
        self.fs = fs_hz
        self.finger_ir_min = finger_ir_min
        self.rr_jump_reject = float(rr_jump_reject)

        self.ir = deque()        # (t, ir)
        self.ir_filt = deque()   # (t, y)
        self.peaks = deque()     # peak times
        self.rr = deque()        # (t, rr)

        self.last_peak_t = None
        self._ma = deque(maxlen=5)

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

            window_raw = [v for (tt, v) in self.ir if (t - tt) <= 2.0]
            med = median(window_raw) if window_raw else ir
            y = ir - (med if med is not None else ir)

            self._ma.append(y)
            y_sm = sum(self._ma) / len(self._ma)

            self.ir_filt.append((t, y_sm))
            self._detect_peak(t)

        self._trim(now_t)

    def _accept_rr(self, rr: float) -> bool:
        if rr < 0.28 or rr > 2.00:
            return False

        recent = [v for (_, v) in list(self.rr)[-8:]]
        if len(recent) >= 5:
            med = median(recent)
            if med and med > 0:
                if abs(rr - med) / med > self.rr_jump_reject:
                    return False
        return True

    def _detect_peak(self, t: float):
        if len(self.ir_filt) < 5:
            return

        (t1, y1) = self.ir_filt[-3]
        (t2, y2) = self.ir_filt[-2]
        (t3, y3) = self.ir_filt[-1]

        if not (y2 > y1 and y2 > y3):
            return

        if self.last_peak_t is not None and (t2 - self.last_peak_t) < 0.25:
            return

        recent = [y for (tt, y) in self.ir_filt if (t2 - tt) <= 5.0]
        if len(recent) < 20:
            return

        m = median(recent)
        d = mad(recent)
        if d is None or d < 2.0:
            return

        thr = (m if m is not None else 0.0) + 0.6 * d
        if y2 < thr:
            return

        self.peaks.append(t2)
        if self.last_peak_t is not None:
            rr = t2 - self.last_peak_t
            if self._accept_rr(rr):
                self.rr.append((t2, rr))

                inst_bpm = 60.0 / rr
                tau = self.bpm_tau if self.bpm_tau > 0 else 1.5
                alpha = 1.0 - math.exp(-rr / tau)
                if self.bpm_ema is None:
                    self.bpm_ema = inst_bpm
                else:
                    self.bpm_ema += alpha * (inst_bpm - self.bpm_ema)

        self.last_peak_t = t2

    def bpm(self):
        if not self.rr:
            return None
        return self.bpm_ema

    def rmssd_30s(self, min_rr: int = 12):
        if not self.rr:
            return None
        now_t = self.rr[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 30.0]
        if len(rrs) < max(3, int(min_rr)):
            return None
        diffs = [(rrs[i] - rrs[i-1]) for i in range(1, len(rrs))]
        sq = [d*d for d in diffs]
        return (sum(sq) / len(sq)) ** 0.5

    def signal_quality(self):
        if not self.finger_on():
            return 0.0
        if not self.rr:
            return 0.0
        now_t = self.rr[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 10.0]
        if len(rrs) < 4:
            return 0.2
        rr_std = stdev(rrs)
        q = 1.0 - norm_range(rr_std, 0.02, 0.20)
        return clamp01(q)


# ----------------------------
# Output-level robust suppressor
# ----------------------------
class RobustSuppressor:
    """
    중앙값/MAD 기반으로 '한 번 튄 값'을 억제.
    - outlier가 1회만 발생하면 median으로 대체
    - 같은 방향(outlier)이 연속 allow_after회 이상이면 "진짜 변화"로 간주하고 통과
    """
    def __init__(self, window=7, k=3.0, abs_floor=0.0, allow_after=2):
        self.buf = deque(maxlen=int(window))
        self.k = float(k)
        self.abs_floor = float(abs_floor)
        self.allow_after = int(allow_after)
        self.streak = 0
        self.last_dir = 0  # -1, 0, +1

    def update(self, x):
        if x is None:
            return None

        if len(self.buf) < 4:
            self.buf.append(float(x))
            self.streak = 0
            self.last_dir = 0
            return float(x)

        med = median(list(self.buf))
        thr = robust_threshold_from_mad(list(self.buf), k=self.k, abs_floor=self.abs_floor)

        diff = float(x) - (med if med is not None else float(x))
        is_out = abs(diff) > thr

        if not is_out:
            self.streak = 0
            self.last_dir = 0
            y = float(x)
        else:
            direction = 1 if diff > 0 else -1
            if direction == self.last_dir:
                self.streak += 1
            else:
                self.streak = 1
                self.last_dir = direction

            if self.streak < self.allow_after:
                # suppress single spike
                y = float(med)
            else:
                # sustained change => pass through
                y = float(x)

        self.buf.append(y)
        return y


# ----------------------------
# BIO Engine v2.6 main
# ----------------------------
def main():
    # ---- config ----
    PPG_FS = 100
    READ_HZ = 50.0
    OUT_HZ = 1.0

    IR_FINGER_MIN = 50000

    # BPM / HRV smoothing
    BPM_EMA_TAU = 1.5
    RR_JUMP_REJECT = 0.20

    RMSSD_MIN_RR = 12
    RMSSD_EMA_TAU = 6.0

    # 출력 튐 억제(robust)
    FREEZE_ON_LOW_PPGQ = True
    PPGQ_FREEZE_THR = 0.45  # 이보다 낮으면 BPM/RMSSD를 "hold"
    BPM_SUPPRESS = RobustSuppressor(window=7, k=3.0, abs_floor=18.0, allow_after=2)       # bpm
    RMSSD_SUPPRESS = RobustSuppressor(window=7, k=3.0, abs_floor=0.050, allow_after=2)    # seconds

    # GSR median filter
    GSR_MEDIAN_N = 3

    # arousal smoothing (fast attack / slow release)
    AROUSAL_SMOOTH = True
    AROUSAL_ATTACK_TAU = 0.8
    AROUSAL_RELEASE_TAU = 2.5

    # normalization ranges (heuristic)
    BPM_LO, BPM_HI = 55, 120
    RMSSD_LO, RMSSD_HI = 0.01, 0.12
    GSR_SLOPE_LO, GSR_SLOPE_HI = 0, 20
    GSR_PEAK_THR = 10.0

    # ---- log dir ----
    base = os.path.join(os.getcwd(), "logs", "bio", time.strftime("%Y%m%d"))
    os.makedirs(base, exist_ok=True)
    csv_path = os.path.join(base, f"bio_v2p6_{time.strftime('%H%M%S')}.csv")

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

    ppg = PPGProcessor(
        fs_hz=PPG_FS,
        finger_ir_min=IR_FINGER_MIN,
        bpm_tau_s=BPM_EMA_TAU,
        rr_jump_reject=RR_JUMP_REJECT,
    )

    # ---- GSR buffers (filtered) ----
    gsr_raw_recent = deque(maxlen=GSR_MEDIAN_N)
    gsr_buf = deque()        # (t, gsr_filtered)
    gsr_slope_buf = deque()  # (t, slope_1s)

    def trim_buf(buf, now_t, secs):
        while buf and (now_t - buf[0][0]) > secs:
            buf.popleft()

    # ---- RMSSD EMA state ----
    rmssd_ema = None
    rmssd_ema_last_t = None

    # ---- arousal smooth state ----
    arousal_sm = None
    last_arousal_t = None

    # ---- hold states ----
    bpm_hold = None
    rmssd_hold = None

    # ---- CSV ----
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts_iso","ts_unix","red","ir",
        "bpm_raw","bpm_filt",
        "rmssd_raw","rmssd_ema","rmssd_filt",
        "gsr_raw","gsr_filt","gsr_slope_1s","gsr_peak_10s",
        "arousal_raw","arousal_filt",
        "bio_conf","ppgQ","finger_on","gsr_fresh","note"
    ])
    f.flush()
    print(f"[INFO] Logging to {csv_path}")

    next_read = time.monotonic()
    next_out = time.monotonic()
    last_red, last_ir = (None, None)

    try:
        while True:
            now_t = time.time()

            # ---- read MAX30102 ----
            samples = max3.read_samples(max_samples=32)
            if samples:
                last_red, last_ir = samples[-1]
                ppg.add_samples(samples, now_t=now_t)

            # ---- read GSR ----
            gsr_val = gsr.latest_value
            gsr_fresh = (gsr.latest_ts is not None and (now_t - gsr.latest_ts) <= 1.0)

            if gsr_val is not None:
                gsr_raw_recent.append(int(gsr_val))
                gsr_filt = median(list(gsr_raw_recent))
                gsr_buf.append((now_t, int(gsr_filt) if gsr_filt is not None else int(gsr_val)))
                trim_buf(gsr_buf, now_t, 10.0)

            # ---- output tick ----
            if time.monotonic() >= next_out:
                next_out += 1.0 / OUT_HZ

                finger_on = ppg.finger_on()
                ppg_q = ppg.signal_quality()

                bpm_raw = ppg.bpm()
                rmssd_raw = ppg.rmssd_30s(min_rr=RMSSD_MIN_RR)

                # RMSSD EMA smoothing
                if rmssd_raw is not None:
                    if rmssd_ema is None:
                        rmssd_ema = float(rmssd_raw)
                        rmssd_ema_last_t = now_t
                    else:
                        dt = max(1e-3, now_t - (rmssd_ema_last_t if rmssd_ema_last_t is not None else now_t))
                        alpha = 1.0 - math.exp(-dt / RMSSD_EMA_TAU)
                        rmssd_ema = rmssd_ema + alpha * (float(rmssd_raw) - rmssd_ema)
                        rmssd_ema_last_t = now_t

                rmssd_use = rmssd_ema if rmssd_ema is not None else rmssd_raw

                # ---- low-quality hold (prevents spikes from motion/noise) ----
                note = ""
                if not finger_on:
                    note = "finger_off"
                elif rmssd_use is None:
                    note = "hrv_warming_up(need~30s)"

                if FREEZE_ON_LOW_PPGQ and ppg_q < PPGQ_FREEZE_THR:
                    # hold previous good values
                    if bpm_hold is not None:
                        bpm_f = bpm_hold
                    else:
                        bpm_f = bpm_raw
                    if rmssd_hold is not None:
                        rmssd_f = rmssd_hold
                    else:
                        rmssd_f = rmssd_use
                    note = (note + "|hold_low_ppgQ").strip("|")
                else:
                    bpm_f = BPM_SUPPRESS.update(bpm_raw) if bpm_raw is not None else None
                    rmssd_f = RMSSD_SUPPRESS.update(rmssd_use) if rmssd_use is not None else None
                    if bpm_f is not None:
                        bpm_hold = bpm_f
                    if rmssd_f is not None:
                        rmssd_hold = rmssd_f

                # ---- GSR features (filtered) ----
                gsr_raw_out = int(gsr_val) if gsr_val is not None else None
                gsr_cur = gsr_buf[-1][1] if gsr_buf else None

                gsr_slope_1s = None
                if gsr_buf:
                    target_t = now_t - 1.0
                    past = None
                    for (t, v) in reversed(gsr_buf):
                        if t <= target_t:
                            past = v
                            break
                    gsr_slope_1s = float(gsr_cur - past) if (past is not None and gsr_cur is not None) else 0.0

                if gsr_slope_1s is not None:
                    gsr_slope_buf.append((now_t, gsr_slope_1s))
                    trim_buf(gsr_slope_buf, now_t, 10.0)

                gsr_peak_10s = sum(1 for (_, s) in gsr_slope_buf if s is not None and s > GSR_PEAK_THR)

                # ---- confidence (separate) ----
                conf = 1.0
                if not finger_on:
                    conf *= 0.0
                if not gsr_fresh:
                    conf *= 0.4
                conf *= (0.3 + 0.7 * ppg_q)
                bio_conf = clamp01(conf)

                # ---- arousal components (use filtered bpm/rmssd) ----
                bpm_c = norm_range(bpm_f, BPM_LO, BPM_HI) if bpm_f is not None else None

                rmssd_inv = None
                if rmssd_f is not None:
                    rmssd_inv = 1.0 - norm_range(rmssd_f, RMSSD_LO, RMSSD_HI)

                gsr_slope_c = norm_range(abs(gsr_slope_1s) if gsr_slope_1s is not None else None,
                                         GSR_SLOPE_LO, GSR_SLOPE_HI)
                gsr_peak_c = norm_range(gsr_peak_10s, 0, 10)

                # v2 scoring 그대로
                if rmssd_inv is None and bpm_c is None:
                    arousal_raw = 0.70 * gsr_slope_c + 0.30 * gsr_peak_c
                elif rmssd_inv is None:
                    arousal_raw = 0.45 * bpm_c + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                else:
                    if bpm_c is None:
                        arousal_raw = 0.45 * rmssd_inv + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                    else:
                        arousal_raw = 0.30 * bpm_c + 0.40 * rmssd_inv + 0.20 * gsr_slope_c + 0.10 * gsr_peak_c

                arousal_raw = clamp01(arousal_raw)

                # optional arousal smoothing: fast rise, slow decay
                if not AROUSAL_SMOOTH:
                    arousal_f = arousal_raw
                else:
                    if arousal_sm is None or last_arousal_t is None:
                        arousal_sm = arousal_raw
                        last_arousal_t = now_t
                    else:
                        dt = max(1e-3, now_t - last_arousal_t)
                        tau = AROUSAL_ATTACK_TAU if arousal_raw > arousal_sm else AROUSAL_RELEASE_TAU
                        alpha = 1.0 - math.exp(-dt / tau)
                        arousal_sm = arousal_sm + alpha * (arousal_raw - arousal_sm)
                        last_arousal_t = now_t
                    arousal_f = clamp01(arousal_sm)

                # ---- CSV ----
                w.writerow([
                    now_iso(), f"{now_t:.3f}",
                    last_red if last_red is not None else "",
                    last_ir if last_ir is not None else "",
                    f"{bpm_raw:.1f}" if bpm_raw is not None else "",
                    f"{bpm_f:.1f}" if bpm_f is not None else "",
                    f"{rmssd_raw:.3f}" if rmssd_raw is not None else "",
                    f"{rmssd_ema:.3f}" if rmssd_ema is not None else "",
                    f"{rmssd_f:.3f}" if rmssd_f is not None else "",
                    gsr_raw_out if gsr_raw_out is not None else "",
                    gsr_cur if gsr_cur is not None else "",
                    f"{gsr_slope_1s:.1f}" if gsr_slope_1s is not None else "",
                    int(gsr_peak_10s),
                    f"{arousal_raw:.2f}",
                    f"{arousal_f:.2f}",
                    f"{bio_conf:.2f}",
                    f"{ppg_q:.2f}",
                    int(finger_on),
                    int(gsr_fresh),
                    note
                ])
                f.flush()

                # ---- terminal print ----
                bpm_s = f"{bpm_f:.1f}" if bpm_f is not None else "..."
                rmssd_s = f"{rmssd_f:.3f}" if rmssd_f is not None else "..."
                gsr_s = f"{gsr_cur}" if gsr_cur is not None else "..."
                print(
                    f"BPM={bpm_s} RMSSD={rmssd_s}s GSR={gsr_s} "
                    f"arousal={arousal_f:.2f} conf={bio_conf:.2f} "
                    f"ppgQ={ppg_q:.2f} note={note}"
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
