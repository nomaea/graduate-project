import math
from collections import deque

from bio_utils import median, mad, stdev, norm_range, clamp01


class PPGProcessor:
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

        self.ir = deque()
        self.ir_filt = deque()
        self.peaks = deque()
        self.rr = deque()

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
        diffs = [(rrs[i] - rrs[i - 1]) for i in range(1, len(rrs))]
        sq = [d * d for d in diffs]
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
