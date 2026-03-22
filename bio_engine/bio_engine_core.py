import os
import time
import csv
import math
from collections import deque

from max30102_better import MAX30102

from bio_utils import clamp01, norm_range, now_iso, mean
from bio_gsr_reader import GSRSerialReader
from bio_ppg_processor import PPGProcessor


def main(payload_queue=None):
    PPG_FS = 100
    READ_HZ = 50.0
    OUT_HZ = 1.0

    IR_FINGER_MIN = 50000
    BPM_EMA_TAU = 1.5
    RR_JUMP_REJECT = 0.20

    RMSSD_MIN_RR = 12
    RMSSD_EMA_TAU = 6.0

    BPM_LO, BPM_HI = 55, 120
    RMSSD_LO, RMSSD_HI = 0.01, 0.12

    GSR_SLOPE_LO, GSR_SLOPE_HI = 0, 20
    GSR_PEAK_THR = 10.0

    base = os.path.join(os.getcwd(), "logs", "bio", time.strftime("%Y%m%d"))
    os.makedirs(base, exist_ok=True)
    csv_path = os.path.join(base, f"bio_v2p5_{time.strftime('%H%M%S')}.csv")

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

    gsr_buf = deque()
    gsr_slope_buf = deque()

    rmssd_ema = None
    rmssd_ema_last_t = None

    def trim_buf(buf, now_t, secs):
        while buf and (now_t - buf[0][0]) > secs:
            buf.popleft()

    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow([
        "ts_iso", "ts_unix", "red", "ir",
        "bpm", "rmssd_raw_s", "rmssd_ema_s", "rr_count_30s",
        "gsr", "gsr_mean_5s", "gsr_slope_1s", "gsr_peak_10s",
        "bio_arousal", "bio_conf", "ppgQ", "finger_on", "gsr_fresh", "note"
    ])
    f.flush()
    print(f"[INFO] Logging to {csv_path}")

    next_read = time.monotonic()
    next_out = time.monotonic()
    last_gsr_ts = None
    last_red, last_ir = (None, None)

    try:
        while True:
            now_t = time.time()

            samples = max3.read_samples(max_samples=32)
            if samples:
                last_red, last_ir = samples[-1]
                ppg.add_samples(samples, now_t=now_t)

            gsr_val = gsr.latest_value
            gsr_ts = gsr.latest_ts
            gsr_fresh = (gsr_ts is not None and (now_t - gsr_ts) <= 1.0)

            if gsr_val is not None and gsr_ts is not None and gsr_ts != last_gsr_ts:
                last_gsr_ts = gsr_ts
                gsr_buf.append((gsr_ts, int(gsr_val)))
                trim_buf(gsr_buf, now_t, 10.0)

            if time.monotonic() >= next_out:
                next_out += 1.0 / OUT_HZ

                finger_on = ppg.finger_on()
                ppg_q = ppg.signal_quality()

                gsr_cur = gsr_buf[-1][1] if gsr_buf else None
                gsr_5s = [v for (t, v) in gsr_buf if (now_t - t) <= 5.0]
                gsr_mean_5s = mean(gsr_5s)

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

                bpm = ppg.bpm()
                rmssd_raw = ppg.rmssd_30s(min_rr=RMSSD_MIN_RR)

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

                conf = 1.0
                if not finger_on:
                    conf *= 0.0
                if not gsr_fresh:
                    conf *= 0.4
                conf *= (0.3 + 0.7 * ppg_q)
                bio_conf = clamp01(conf)

                bpm_c = norm_range(bpm, BPM_LO, BPM_HI) if bpm is not None else None

                rmssd_inv = None
                if rmssd_use is not None:
                    rmssd_inv = 1.0 - norm_range(rmssd_use, RMSSD_LO, RMSSD_HI)

                gsr_slope_c = norm_range(abs(gsr_slope_1s) if gsr_slope_1s is not None else None, GSR_SLOPE_LO, GSR_SLOPE_HI)
                gsr_peak_c = norm_range(gsr_peak_10s, 0, 10)

                if rmssd_inv is None and bpm_c is None:
                    arousal = 0.70 * gsr_slope_c + 0.30 * gsr_peak_c
                elif rmssd_inv is None:
                    arousal = 0.45 * bpm_c + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                else:
                    if bpm_c is None:
                        arousal = 0.45 * rmssd_inv + 0.35 * gsr_slope_c + 0.20 * gsr_peak_c
                    else:
                        arousal = 0.30 * bpm_c + 0.40 * rmssd_inv + 0.20 * gsr_slope_c + 0.10 * gsr_peak_c

                bio_arousal = clamp01(arousal)

                note = ""
                if not finger_on:
                    note = "finger_off"
                elif rmssd_use is None:
                    note = "hrv_warming_up(need~30s)"

                rr_count_30s = 0
                if ppg.rr:
                    t_last = ppg.rr[-1][0]
                    rr_count_30s = sum(1 for (tt, _) in ppg.rr if (t_last - tt) <= 30.0)

                w.writerow([
                    now_iso(), f"{now_t:.3f}",
                    last_red if last_red is not None else "",
                    last_ir if last_ir is not None else "",
                    f"{bpm:.1f}" if bpm is not None else "",
                    f"{rmssd_raw:.3f}" if rmssd_raw is not None else "",
                    f"{rmssd_ema:.3f}" if rmssd_ema is not None else "",
                    rr_count_30s,
                    gsr_cur if gsr_cur is not None else "",
                    f"{gsr_mean_5s:.1f}" if gsr_mean_5s is not None else "",
                    f"{gsr_slope_1s:.1f}" if gsr_slope_1s is not None else "",
                    int(gsr_peak_10s),
                    f"{bio_arousal:.1f}",
                    f"{bio_conf:.2f}",
                    f"{ppg_q:.2f}",
                    int(finger_on),
                    int(gsr_fresh),
                    note
                ])
                f.flush()

                payload = {
                    "timestamp": now_iso(),
                    "bio_arousal": float(bio_arousal),
                    "bio_conf": float(bio_conf),
                    "bpm": float(bpm) if bpm is not None else None,
                    "rmssd_use": float(rmssd_use) if rmssd_use is not None else None,
                    "gsr_cur": float(gsr_cur) if gsr_cur is not None else None,
                    "gsr_slope_1s": float(gsr_slope_1s) if gsr_slope_1s is not None else None,
                    "gsr_peak_10s": int(gsr_peak_10s),
                    "ppgQ": float(ppg_q),
                    "finger_on": bool(finger_on),
                    "gsr_fresh": bool(gsr_fresh),
                    "note": note,
                }

                if payload_queue is not None:
                    try:
                        if payload_queue.full():
                            payload_queue.get_nowait()
                        payload_queue.put_nowait(payload)
                    except Exception:
                        pass

                bpm_s = f"{bpm:.1f}" if bpm is not None else "..."
                rmssd_s = f"{rmssd_use:.3f}" if rmssd_use is not None else "..."
                gsr_s = f"{gsr_cur}" if gsr_cur is not None else "..."

                print(
                    f"BPM={bpm_s} RMSSD={rmssd_s}s GSR={gsr_s} "
                    f"arousal={bio_arousal:.1f} conf={bio_conf:.2f} "
                    f"ppgQ={ppg_q:.2f} finger_on={int(finger_on)} note={note}"
                )

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
