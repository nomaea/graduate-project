
# bio_engine_v3_baseline.py
# MAX30102 (I2C) + Arduino GSR (Serial) -> BPM + HRV(RMSSD) + EDA phasic -> baseline-normalized arousal (0~1)
#
# 주요 개선점(v2 대비)
# - 개인 baseline(초기 안정상태) 기반 z-score 표준화
# - RMSSD는 lnRMSSD 사용 + RR 아티팩트 제거 + EMA 스무딩으로 튐 감소
# - EDA(GSR)는 tonic/phasic 분리(EMA tonic) 후 phasic RMS/피크율 사용 (slope 근사보다 안정적)
# - arousal=σ(S) 형태(0~1), calm=1-arousal
#
# 필요 패키지:
#   pip install smbus2 pyserial
#
import os
import time
import csv
import re
import math
import json
import threading
from collections import deque

import serial
from serial.tools import list_ports

from max30102_better import MAX30102


# ----------------------------
# Utils
# ----------------------------
def clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def now_iso():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def stdev(xs):
    if xs is None or len(xs) < 2:
        return None
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
    # median absolute deviation
    if not xs:
        return None
    m = median(xs)
    dev = [abs(x - m) for x in xs]
    return median(dev)

def robust_z(x, med, mad_val, eps=1e-9):
    """
    robust z-score using MAD:
      z = (x - median) / (1.4826 * MAD)
    If MAD is too small -> return 0.
    """
    if x is None or med is None or mad_val is None:
        return 0.0
    denom = 1.4826 * mad_val
    if denom < eps:
        return 0.0
    return (x - med) / denom

def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


# ----------------------------
# GSR Serial Reader
# ----------------------------
class GSRSerialReader:
    """
    Arduino line examples:
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
# PPG Processor (BPM + RR + HRV) with artifact rejection
# ----------------------------
class PPGProcessor:
    def __init__(self, fs_hz: int, finger_ir_min: int = 50000, bpm_tau_s: float = 1.5):
        self.fs = fs_hz
        self.finger_ir_min = finger_ir_min

        self.ir = deque()        # (t, ir)
        self.ir_filt = deque()   # (t, y)
        self.peaks = deque()     # peak times

        # RR intervals (clean)
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

    def _accept_rr(self, rr: float) -> bool:
        # physiological sanity
        if rr < 0.28 or rr > 2.00:
            return False

        # artifact rejection: if we have enough RR history, reject large jumps
        # (PPG peak jitter/false peak causes huge rr change -> RMSSD spikes)
        recent = [v for (_, v) in list(self.rr)[-8:]]
        if len(recent) >= 5:
            med = median(recent)
            if med and med > 0:
                if abs(rr - med) / med > 0.20:  # >20% jump -> likely artifact
                    return False
        return True

    def _detect_peak(self, t: float):
        if len(self.ir_filt) < 5:
            return

        (t1, y1) = self.ir_filt[-3]
        (t2, y2) = self.ir_filt[-2]
        (t3, y3) = self.ir_filt[-1]

        # local max at middle
        if not (y2 > y1 and y2 > y3):
            return

        # refractory to avoid double-peaks (0.25s)
        if self.last_peak_t is not None and (t2 - self.last_peak_t) < 0.25:
            return

        # adaptive threshold based on last 5s
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

        # accept peak
        self.peaks.append(t2)
        if self.last_peak_t is not None:
            rr = t2 - self.last_peak_t
            if self._accept_rr(rr):
                self.rr.append((t2, rr))

                # EMA BPM update
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

    def rmssd_30s(self):
        if not self.rr:
            return None
        now_t = self.rr[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 30.0]
        # 안정적인 RMSSD를 위해 최소 개수 요구(대략 30초면 30~40개)
        if len(rrs) < 12:
            return None
        diffs = [(rrs[i] - rrs[i-1]) for i in range(1, len(rrs))]
        sq = [d*d for d in diffs]
        return (sum(sq) / len(sq)) ** 0.5

    def signal_quality(self):
        # based on RR stability
        if not self.finger_on():
            return 0.0
        if not self.rr:
            return 0.0
        now_t = self.rr[-1][0]
        rrs = [rr for (tt, rr) in self.rr if (now_t - tt) <= 10.0]
        if len(rrs) < 4:
            return 0.2
        rr_std = stdev(rrs)
        if rr_std is None:
            return 0.2
        # rr_std small -> quality high
        q = 1.0 - clamp01((rr_std - 0.02) / (0.20 - 0.02))
        return clamp01(q)


# ----------------------------
# Baseline calibrator (robust stats)
# ----------------------------
class BaselineCalibrator:
    def __init__(self, target_good_seconds: int = 60):
        self.target = int(target_good_seconds)
        self.good = 0
        self.hr = []
        self.lnrmssd = []
        self.eda = []
        self.ready = False
        self.stats = {}

    def update(self, ok: bool, hr: float, rmssd: float, eda_feat: float):
        if self.ready:
            return
        if not ok:
            return

        # HR
        if hr is not None:
            self.hr.append(float(hr))
        # lnRMSSD
        if rmssd is not None and rmssd > 0:
            self.lnrmssd.append(math.log(float(rmssd)))
        # EDA feature
        if eda_feat is not None:
            self.eda.append(float(eda_feat))

        self.good += 1

        # need enough samples in each channel
        if self.good >= self.target and len(self.hr) >= 30 and len(self.lnrmssd) >= 20 and len(self.eda) >= 30:
            self.stats = {
                "hr_med": median(self.hr),
                "hr_mad": mad(self.hr),
                "lnrmssd_med": median(self.lnrmssd),
                "lnrmssd_mad": mad(self.lnrmssd),
                "eda_med": median(self.eda),
                "eda_mad": mad(self.eda),
                "good_seconds": self.good,
                "target_good_seconds": self.target,
            }
            self.ready = True

    def progress(self):
        return self.good, self.target


# ----------------------------
# Main
# ----------------------------
def main():
    # ---- config ----
    PPG_FS = 100
    READ_HZ = 50.0       # read loop (avoid FIFO overflow)
    OUT_HZ = 1.0         # output/log loop
    BPM_EMA_TAU = 1.5
    RMSSD_EMA_TAU = 6.0  # smooth RMSSD for stability (seconds)
    IR_FINGER_MIN = 50000

    BASELINE_GOOD_SEC = 60  # require this many "good" seconds for baseline

    # EDA processing
    EDA_TONIC_TAU = 10.0      # seconds, tonic EMA time constant
    EDA_PEAK_K = 2.0          # threshold = median + K*MAD (phasic window)
    EDA_PEAK_MIN = 1.0        # minimum peak threshold in raw units (avoid 0 threshold)
    EDA_PHASIC_WIN = 10.0     # seconds

    # Arousal weights (sum to 1.0)
    W_HR = 0.25
    W_HRV = 0.50  # (-z_lnrmssd)
    W_EDA = 0.25

    # ---- log dir ----
    base = os.path.join(os.getcwd(), "logs", "bio", time.strftime("%Y%m%d"))
    os.makedirs(base, exist_ok=True)
    session_id = time.strftime("%H%M%S")
    csv_path = os.path.join(base, f"bio_v3_{session_id}.csv")
    baseline_path = os.path.join(base, f"baseline_{session_id}.json")

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
    baseline = BaselineCalibrator(target_good_seconds=BASELINE_GOOD_SEC)

    # GSR buffers
    gsr_buf = deque()       # (t, raw)
    phasic_buf = deque()    # (t, phasic)
    last_gsr_ts = None
    tonic = None
    tonic_last_ts = None

    # RMSSD smoothing
    rmssd_ema = None
    rmssd_ema_last_t = None

    def trim_buf(buf, now_t, secs):
        while buf and (now_t - buf[0][0]) > secs:
            buf.popleft()

    # CSV
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts_iso","ts_unix","red","ir",
        "bpm","rmssd","rmssd_ema",
        "gsr","gsr_mean_5s","gsr_slope_1s",
        "eda_tonic","eda_phasic_rms_10s","eda_scr_rate_per_min",
        "baseline_ready","z_hr","z_lnrmssd","z_eda","S","arousal_rel","arousal","calm",
        "bio_conf","ppgQ","finger_on","gsr_fresh","note"
    ])
    f.flush()
    print(f"[INFO] Logging to {csv_path}")
    print("[INFO] Baseline: keep still for ~60 good seconds (finger on, stable).")

    next_read = time.monotonic()
    next_out = time.monotonic()
    last_red, last_ir = (None, None)

    try:
        while True:
            now_t = time.time()

            # ---- fast read MAX30102 ----
            samples = max3.read_samples(max_samples=32)
            if samples:
                last_red, last_ir = samples[-1]
                ppg.add_samples(samples, now_t=now_t)

            # ---- update GSR buffer on new reading only ----
            gsr_val = gsr.latest_value
            gsr_ts = gsr.latest_ts
            gsr_fresh = (gsr_ts is not None and (now_t - gsr_ts) <= 1.0)

            if gsr_val is not None and gsr_ts is not None and gsr_ts != last_gsr_ts:
                last_gsr_ts = gsr_ts
                gsr_buf.append((gsr_ts, int(gsr_val)))
                trim_buf(gsr_buf, now_t, 20.0)

                # tonic/phasic update
                if tonic is None:
                    tonic = float(gsr_val)
                    tonic_last_ts = gsr_ts
                else:
                    dt = max(1e-3, float(gsr_ts - (tonic_last_ts if tonic_last_ts is not None else gsr_ts)))
                    alpha = 1.0 - math.exp(-dt / EDA_TONIC_TAU)
                    tonic = tonic + alpha * (float(gsr_val) - tonic)
                    tonic_last_ts = gsr_ts

                phasic = float(gsr_val) - float(tonic)
                # SCR is usually positive-going; keep positive part to reduce drift effects
                phasic_pos = max(0.0, phasic)
                phasic_buf.append((gsr_ts, phasic_pos))
                trim_buf(phasic_buf, now_t, 20.0)

            # ---- output/log tick ----
            if time.monotonic() >= next_out:
                next_out += 1.0 / OUT_HZ

                finger_on = ppg.finger_on()
                ppg_q = ppg.signal_quality()

                # HR/HRV
                bpm = ppg.bpm()
                rmssd = ppg.rmssd_30s()

                # RMSSD EMA smoothing (reduce jitter)
                if rmssd is not None:
                    if rmssd_ema is None:
                        rmssd_ema = float(rmssd)
                        rmssd_ema_last_t = now_t
                    else:
                        dt = max(1e-3, now_t - (rmssd_ema_last_t if rmssd_ema_last_t is not None else now_t))
                        alpha = 1.0 - math.exp(-dt / RMSSD_EMA_TAU)
                        rmssd_ema = rmssd_ema + alpha * (float(rmssd) - rmssd_ema)
                        rmssd_ema_last_t = now_t

                # GSR mean 5s
                gsr_5s = [v for (t, v) in gsr_buf if (now_t - t) <= 5.0]
                gsr_mean_5s = mean(gsr_5s)
                gsr_cur = gsr_buf[-1][1] if gsr_buf else None

                # slope 1s (difference vs nearest <=1s ago)
                gsr_slope_1s = None
                if gsr_buf:
                    target_t = now_t - 1.0
                    past = None
                    for (t, v) in reversed(gsr_buf):
                        if t <= target_t:
                            past = v
                            break
                    gsr_slope_1s = float(gsr_cur - past) if (past is not None and gsr_cur is not None) else 0.0

                # EDA phasic RMS in last 10s
                ph_10 = [v for (t, v) in phasic_buf if (now_t - t) <= EDA_PHASIC_WIN]
                eda_phasic_rms = None
                if ph_10:
                    eda_phasic_rms = math.sqrt(sum(v*v for v in ph_10) / len(ph_10))

                # EDA SCR rate (events/min) using simple peak count in last 10s
                scr_rate = None
                if len(ph_10) >= 5:
                    m = median(ph_10) or 0.0
                    d = mad(ph_10) or 0.0
                    thr = max(EDA_PEAK_MIN, m + EDA_PEAK_K * d)
                    peaks = 0
                    # local maxima in ph_10 sequence
                    for i in range(1, len(ph_10) - 1):
                        if ph_10[i] > thr and ph_10[i] > ph_10[i-1] and ph_10[i] > ph_10[i+1]:
                            peaks += 1
                    # convert (peaks per 10s) -> per minute
                    scr_rate = peaks * (60.0 / EDA_PHASIC_WIN)

                # confidence (keep separate)
                conf = 1.0
                if not finger_on:
                    conf *= 0.0
                if not gsr_fresh:
                    conf *= 0.4
                conf *= (0.3 + 0.7 * ppg_q)
                bio_conf = clamp01(conf)

                # baseline update 조건(좋은 샘플)
                ok_for_baseline = (
                    finger_on and gsr_fresh and ppg_q >= 0.6 and
                    (bpm is not None) and (rmssd_ema is not None) and (eda_phasic_rms is not None)
                )
                baseline.update(ok_for_baseline, bpm, rmssd_ema, eda_phasic_rms)

                # compute arousal
                baseline_ready = baseline.ready
                z_hr = z_lnrmssd = z_eda = 0.0
                S = 0.0
                arousal_rel = None
                arousal = None
                calm = None

                note = ""
                if not finger_on:
                    note = "finger_off"
                elif rmssd_ema is None:
                    note = "hrv_warming_up(need~30s)"
                elif not baseline_ready:
                    g, t = baseline.progress()
                    note = f"calibrating_baseline({g}/{t})"

                if baseline_ready:
                    stats = baseline.stats

                    lnrmssd = math.log(rmssd_ema) if (rmssd_ema is not None and rmssd_ema > 0) else None
                    z_hr = robust_z(bpm, stats["hr_med"], stats["hr_mad"])
                    z_lnrmssd = robust_z(lnrmssd, stats["lnrmssd_med"], stats["lnrmssd_mad"])
                    z_eda = robust_z(eda_phasic_rms, stats["eda_med"], stats["eda_mad"])

                    # clip extreme z to reduce spikes
                    z_hr = clamp(z_hr, -3.0, 3.0)
                    z_lnrmssd = clamp(z_lnrmssd, -3.0, 3.0)
                    z_eda = clamp(z_eda, -3.0, 3.0)

                    # weight gating by signal availability/quality
                    w_hr = W_HR * ppg_q
                    w_hrv = W_HRV * ppg_q
                    w_eda = W_EDA * (1.0 if gsr_fresh else 0.0)
                    w_sum = (w_hr + w_hrv + w_eda) if (w_hr + w_hrv + w_eda) > 1e-6 else 1.0

                    S = (w_hr * z_hr + w_hrv * (-z_lnrmssd) + w_eda * z_eda) / w_sum

                    # arousal_rel: baseline≈0.5 (S≈0), arousal: above-baseline (0~1)
                    arousal_rel = sigmoid(S)
                    arousal = clamp01(2.0 * (arousal_rel - 0.5))
                    calm = 1.0 - arousal
                # write CSV
                w.writerow([
                    now_iso(), f"{now_t:.3f}",
                    last_red if last_red is not None else "",
                    last_ir if last_ir is not None else "",
                    f"{bpm:.1f}" if bpm is not None else "",
                    f"{rmssd:.3f}" if rmssd is not None else "",
                    f"{rmssd_ema:.3f}" if rmssd_ema is not None else "",
                    gsr_cur if gsr_cur is not None else "",
                    f"{gsr_mean_5s:.1f}" if gsr_mean_5s is not None else "",
                    f"{gsr_slope_1s:.1f}" if gsr_slope_1s is not None else "",
                    f"{tonic:.1f}" if tonic is not None else "",
                    f"{eda_phasic_rms:.2f}" if eda_phasic_rms is not None else "",
                    f"{scr_rate:.1f}" if scr_rate is not None else "",
                    int(baseline_ready),
                    f"{z_hr:.2f}",
                    f"{z_lnrmssd:.2f}",
                    f"{z_eda:.2f}",
                    f"{S:.2f}",
                    f"{arousal_rel:.2f}" if arousal_rel is not None else "",
                    f"{arousal:.2f}" if arousal is not None else "",
                    f"{calm:.2f}" if calm is not None else "",
                    f"{bio_conf:.2f}",
                    f"{ppg_q:.2f}",
                    int(finger_on),
                    int(gsr_fresh),
                    note
                ])
                f.flush()

                # save baseline once when ready
                if baseline_ready and not os.path.exists(baseline_path):
                    try:
                        with open(baseline_path, "w", encoding="utf-8") as jf:
                            json.dump(baseline.stats, jf, ensure_ascii=False, indent=2)
                        print(f"[INFO] Baseline saved to {baseline_path}")
                    except Exception:
                        pass

                # terminal print
                bpm_s = f"{bpm:.1f}" if bpm is not None else "..."
                rmssd_s = f"{rmssd_ema:.3f}" if rmssd_ema is not None else "..."
                gsr_s = f"{gsr_cur}" if gsr_cur is not None else "..."
                ar_rel_s = f"{arousal_rel:.2f}" if arousal_rel is not None else "..."
                ar_s = f"{arousal:.2f}" if arousal is not None else "..."
                calm_s = f"{calm:.2f}" if calm is not None else "..."
                print(
                    f"BPM={bpm_s} RMSSD={rmssd_s}s GSR={gsr_s} "
                    f"arousal={ar_s} (rel={ar_rel_s}) calm={calm_s} conf={bio_conf:.2f} "
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