import logging
import math
import os
import csv
import time
from collections import deque

from .max30102_better import MAX30102
from .bio_utils import clamp01, norm_range, now_iso, mean
from .bio_gsr_reader import GSRSerialReader
from .bio_ppg_processor import PPGProcessor

# FIX #9: print → logging 으로 교체
logger = logging.getLogger(__name__)


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
    # FIX #6: 정규화 기준을 상수로 분리 + 출처 명시
    # 근거: Boucsein (2012) Electrodermal Activity p.194 — 분당 ~20회 피크 ≈ 강한 스트레스
    #       10초 기준으로 환산하면 약 3~5개. 5를 상한으로 설정.
    GSR_PEAK_MAX = 5

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
    # FIX #2: start()가 raise 대신 error 플래그를 설정하므로 여기서 확인
    if gsr.error:
        logger.warning(f"[BIO] GSR unavailable: {gsr.error} — running without GSR")
    else:
        logger.info(f"[BIO] GSR serial port = {gsr.port}")

    ppg = PPGProcessor(
        fs_hz=PPG_FS,
        finger_ir_min=IR_FINGER_MIN,
        bpm_tau_s=BPM_EMA_TAU,
        rr_jump_reject=RR_JUMP_REJECT,
    )

    gsr_buf = deque()
    gsr_slope_buf = deque()

    rmssd_ema = None
    # FIX #4: EMA dt 계산을 monotonic 기준으로 변경 (wall clock은 NTP 보정 시 역방향 점프 가능)
    rmssd_ema_last_mono = None

    def trim_buf(buf, now_t, secs):
        while buf and (now_t - buf[0][0]) > secs:
            buf.popleft()

    # FIX #8: CSV open을 try 바깥으로 분리 — f 미정의 상태에서 finally 진입 방지
    try:
        f = open(csv_path, "w", newline="")
    except OSError as e:
        logger.error(f"[BIO] Cannot open CSV {csv_path}: {e}")
        raise

    try:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "ts_iso", "ts_unix", "red", "ir",
            "bpm", "rmssd_raw_s", "rmssd_ema_s", "rr_count_30s",
            "gsr", "gsr_mean_5s", "gsr_slope_1s", "gsr_peak_10s",
            "bio_arousal", "bio_conf", "ppgQ", "finger_on", "gsr_fresh", "note"
        ])
        f.flush()
        logger.info(f"[BIO] Logging to {csv_path}")

        next_read = time.monotonic()
        next_out = time.monotonic()
        last_gsr_ts = None
        last_red, last_ir = (None, None)

        while True:
            # FIX #4: 타이밍 전용 monotonic / 절대 시각 전용 wall clock 분리
            now_mono = time.monotonic()
            now_wall = time.time()  # CSV 타임스탬프 · GSR 신선도 확인에만 사용

            samples = max3.read_samples(max_samples=32)
            if samples:
                last_red, last_ir = samples[-1]
                ppg.add_samples(samples, now_t=now_wall)

            gsr_val = gsr.latest_value
            gsr_ts = gsr.latest_ts
            gsr_fresh = (gsr_ts is not None and (now_wall - gsr_ts) <= 1.0)

            if gsr_fresh and gsr_val is not None:
                if (last_gsr_ts is None) or (gsr_ts != last_gsr_ts):
                    gsr_buf.append((now_wall, float(gsr_val)))
                    gsr_slope_buf.append((now_wall, float(gsr_val)))
                    last_gsr_ts = gsr_ts

            trim_buf(gsr_buf, now_wall, 10.0)
            trim_buf(gsr_slope_buf, now_wall, 1.0)

            if now_mono >= next_out:
                next_out += 1.0 / OUT_HZ

                bpm = ppg.bpm()
                rmssd_raw = ppg.rmssd_30s()

                # FIX #1: rr_30s property 사용 — 기존 getattr 항상 None 반환 버그 수정
                rr_count = len(ppg.rr_30s)

                ppgQ = ppg.signal_quality()
                finger_on = ppg.finger_on()

                # FIX #4: rmssd EMA dt를 monotonic 기준으로 계산
                if rmssd_raw is not None:
                    if rmssd_ema is None:
                        rmssd_ema = rmssd_raw
                    else:
                        dt = (now_mono - rmssd_ema_last_mono) if rmssd_ema_last_mono is not None else 1.0
                        a = 1.0 - math.exp(-max(dt, 1e-3) / RMSSD_EMA_TAU)
                        rmssd_ema = (1.0 - a) * rmssd_ema + a * rmssd_raw
                    rmssd_ema_last_mono = now_mono

                gsr_cur = float(gsr_val) if gsr_val is not None else None

                gsr_mean_5s = None
                if gsr_buf:
                    vals5 = [v for t, v in gsr_buf if (now_wall - t) <= 5.0]
                    if vals5:
                        gsr_mean_5s = mean(vals5)

                gsr_slope_1s = 0.0
                vals1 = [v for t, v in gsr_slope_buf]
                ts1 = [t for t, v in gsr_slope_buf]
                if len(vals1) >= 2:
                    dt_gsr = max(ts1[-1] - ts1[0], 1e-3)
                    gsr_slope_1s = (vals1[-1] - vals1[0]) / dt_gsr

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

                # FIX #3: None 신호를 0.0으로 fallback하지 않고 가중합에서 제외 후 재분배
                # 기존: bpm=None → bpm_n=0.0 (BPM 최솟값으로 오해됨)
                #       rmssd=None → hrv_n_inv=0.0 (HRV 최댓값 = 스트레스 없음으로 오해됨)
                rmssd_use = rmssd_ema if rmssd_ema is not None else rmssd_raw
                gsr_slope_n = norm_range(gsr_slope_1s, GSR_SLOPE_LO, GSR_SLOPE_HI)
                # FIX #6: 매직넘버 5.0 → GSR_PEAK_MAX 상수 사용
                gsr_peak_n = clamp01(gsr_peak_10s / GSR_PEAK_MAX)

                arousal_terms = []
                if bpm is not None:
                    arousal_terms.append((norm_range(bpm, BPM_LO, BPM_HI), 0.35))
                if rmssd_use is not None:
                    arousal_terms.append((1.0 - norm_range(rmssd_use, RMSSD_LO, RMSSD_HI), 0.30))
                arousal_terms.append((gsr_slope_n, 0.20))
                arousal_terms.append((gsr_peak_n, 0.15))

                total_w = sum(weight for _, weight in arousal_terms)
                bio_arousal = clamp01(
                    sum(val * weight for val, weight in arousal_terms) / total_w
                ) if total_w > 0 else 0.0

                conf = 1.0
                if not finger_on:
                    conf *= 0.1
                conf *= clamp01((ppgQ - 0.2) / 0.6)
                if not gsr_fresh:
                    conf *= 0.5
                bio_conf = clamp01(conf)

                ts_iso = now_iso()
                ts_unix = now_wall

                csv_writer.writerow([
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
                            dropped = payload_queue.get_nowait()
                            logger.debug(f"[BIO] Dropped stale payload ts={dropped.get('timestamp')}")
                        payload_queue.put_nowait(payload)
                        logger.debug(
                            f"[BIO] Payload queued ts={payload['timestamp']} "
                            f"bpm={payload['bpm']} "
                            f"arousal={payload['bio_arousal']:.2f} "
                            f"conf={payload['bio_conf']:.2f} "
                            f"note={payload['note']}"
                        )
                    except Exception as e:
                        logger.warning(f"[BIO][QUEUE] {type(e).__name__}: {e}")

                # FIX #10: bpm=0.0 이 falsy로 평가되는 버그 수정
                # 기존: bpm if bpm else 0  →  bpm=0.0 일 때 0 출력 (정상값인데 None 취급)
                bpm_log = bpm if bpm is not None else 0.0
                rmssd_log = rmssd_use if rmssd_use is not None else 0.0
                gsr_log = gsr_cur if gsr_cur is not None else 0.0
                logger.info(
                    f"[BIO v2.5] BPM={bpm_log:.1f} RMSSD={rmssd_log:.4f} "
                    f"GSR={gsr_log:.1f} Slope1s={gsr_slope_1s:.2f} Peaks10s={gsr_peak_10s} "
                    f"Arousal={bio_arousal:.2f} Conf={bio_conf:.2f} "
                    f"Q={ppgQ:.2f} Finger={int(finger_on)} GSRfresh={int(gsr_fresh)} {note}"
                )

            next_read += 1.0 / READ_HZ
            sleep_s = next_read - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("[BIO] Stopped by user")
    finally:
        try:
            gsr.stop()
        except Exception:
            pass
        try:
            max3.close()
        except Exception:
            pass
        # FIX #8: f는 try 블록 바깥에서 open되므로 항상 정의됨 — NameError 없이 안전하게 닫힘
        f.close()
