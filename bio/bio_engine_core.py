import os
import time
import csv
import math
from collections import deque

from .max30102_better import MAX30102
from .bio_utils import clamp01, norm_range, now_iso, mean
from .bio_gsr_reader import GSRSerialReader
from .bio_ppg_processor import PPGProcessor


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

            if gsr_fresh and gsr_val is not None:
                if (last_gsr_ts is None) or (gsr_ts != last_gsr_ts):
                    gsr_buf.append((now_t, float(gsr_val)))
                    gsr_slope_buf.append((now_t, float(gsr_val)))
                    last_gsr_ts = gsr_ts

            trim_buf(gsr_buf, now_t, 10.0)
            trim_buf(gsr_slope_buf, now_t, 1.0)

            if time.monotonic() >= next_out:
                next_out += 1.0 / OUT_HZ

                bpm = ppg.bpm()
                rmssd_raw = ppg.rmssd_30s()
                rr_count = len(ppg.rr_30s)
                ppgQ = ppg.signal_quality()
                finger_on = ppg.finger_on()

                if rmssd_raw is not None:
                    if rmssd_ema is None:
                        rmssd_ema = rmssd_raw
                    else:
                        dt = (now_t - rmssd_ema_last_t) if rmssd_ema_last_t else 1.0
                        a = 1.0 - math.exp(-max(dt, 1e-3) / RMSSD_EMA_TAU)
                        rmssd_ema = (1.0 - a) * rmssd_ema + a * rmssd_raw
                    rmssd_ema_last_t = now_t

                gsr_cur = float(gsr_val) if gsr_val is not None else None
                gsr_mean_5s = None
                if len(gsr_buf) > 0:
                    vals5 = [v for t, v in gsr_buf if (now_t - t) <= 5.0]
                    if vals5:
                        gsr_mean_5s = mean(vals5)

                gsr_slope_1s = 0.0
                vals1 = [v for t, v in gsr_slope_buf]
                ts1 = [t for t, v in gsr_slope_buf]
                if len(vals1) >= 2:
                    dt = max(ts1[-1] - ts1[0], 1e-3)
                    gsr_slope_1s = (vals1[-1] - vals1[0]) / dt

                gsr_peak_10s = 0
                if len(gsr_buf) >= 3:
                    vals10 = [v for t, v in gsr_buf]
                    for i in range(1, len(vals10) - 1):
                        if vals10[i] > vals10[i - 1] and vals10[i] > vals10[i + 1]:
                            if (vals10[i] - min(vals10[max(0, i - 3):i + 1])) >= GSR_PEAK_THR:
                                gsr_peak_10s += 1

                note = ""
                if not finger_on:
                    note = "NO_FINGER"
                elif ppgQ < 0.35:
                    note = "LOW_PPG_Q"
                elif not gsr_fresh:
                    note = "NO_GSR"

                bpm_n = norm_range(bpm, BPM_LO, BPM_HI) if bpm is not None else 0.0
                rmssd_use = rmssd_ema if rmssd_ema is not None else rmssd_raw
                hrv_n_inv = 1.0 - norm_range(rmssd_use, RMSSD_LO, RMSSD_HI) if rmssd_use is not None else 0.0
                gsr_slope_n = norm_range(gsr_slope_1s, GSR_SLOPE_LO, GSR_SLOPE_HI)
                gsr_peak_n = clamp01(gsr_peak_10s / 5.0)

                bio_arousal = clamp01(
                    0.35 * bpm_n +
                    0.30 * hrv_n_inv +
                    0.20 * gsr_slope_n +
                    0.15 * gsr_peak_n
                )

                conf = 1.0
                if not finger_on:
                    conf *= 0.1
                conf *= clamp01((ppgQ - 0.2) / 0.6)
                if not gsr_fresh:
                    conf *= 0.5
                bio_conf = clamp01(conf)

                ts_iso = now_iso()
                ts_unix = now_t

                w.writerow([
                    ts_iso, f"{ts_unix:.3f}",
                    last_red if last_red is not None else "",
                    last_ir if last_ir is not None else "",
                    f"{bpm:.2f}" if bpm is not None else "",
                    f"{rmssd_raw:.4f}" if rmssd_raw is not None else "",
                    f"{rmssd_ema:.4f}" if rmssd_ema is not None else "",
                    rr_count,
                    f"{gsr_cur:.2f}" if gsr_cur is not None else "",
                    f"{gsr_mean_5s:.2f}" if gsr_mean_5s is not None else "",
                    f"{gsr_slope_1s:.2f}",
                    gsr_peak_10s,
                    f"{bio_arousal:.3f}",
                    f"{bio_conf:.3f}",
                    f"{ppgQ:.3f}",
                    int(finger_on),
                    int(gsr_fresh),
                    note
                ])
                f.flush()

                payload = {
                    "timestamp": ts_iso,
                    "timestamp_unix": ts_unix,
                    "bio_arousal": bio_arousal,
                    "bio_conf": bio_conf,
                    "bpm": bpm,
                    "rmssd_use": rmssd_use,
                    "gsr_cur": gsr_cur,
                    "gsr_slope_1s": gsr_slope_1s,
                    "gsr_peak_10s": gsr_peak_10s,
                    "ppgQ": ppgQ,
                    "finger_on": finger_on,
                    "gsr_fresh": gsr_fresh,
                    "note": note,
                }

                if payload_queue is not None:
                    try:
                        if payload_queue.full():
                            payload_queue.get_nowait()
                        payload_queue.put_nowait(payload)
                    except Exception:
                        pass

                print(
                    f"[BIO v2.5] BPM={bpm if bpm else 0:.1f} "
                    f"RMSSD={rmssd_use if rmssd_use else 0:.4f} "
                    f"GSR={gsr_cur if gsr_cur else 0:.1f} "
                    f"Slope1s={gsr_slope_1s:.2f} Peaks10s={gsr_peak_10s} "
                    f"Arousal={bio_arousal:.2f} Conf={bio_conf:.2f} "
                    f"Q={ppgQ:.2f} Finger={finger_on} GSRfresh={gsr_fresh} {note}"
                )

            next_read += 1.0 / READ_HZ
            sleep_s = next_read - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")
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
