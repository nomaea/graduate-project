import time
from collections import deque
from max30102_better import MAX30102, MODE_SPO2

def estimate_bpm_peaks(sig, fs):
    """
    sig: DC 제거된(평균 제거) IR 신호
    간단/안정 피크 검출 -> BPM 계산
    """
    if len(sig) < int(fs * 4):
        return None

    # smoothing
    win = 7
    sm = []
    dq = deque(maxlen=win)
    for v in sig:
        dq.append(v)
        sm.append(sum(dq) / len(dq))

    # dynamic threshold: 너무 높지 않게(강하게 튀면 피크 못 잡음 방지)
    abs_mean = sum(abs(v) for v in sm) / len(sm)
    thr = abs_mean * 0.8   # 기존 1.2 -> 0.8로 낮춤

    peaks = []
    min_interval = int(fs * 0.33)  # ~180 bpm max
    last = -10**9

    for i in range(2, len(sm) - 2):
        if i - last < min_interval:
            continue
        # local maxima + threshold
        if sm[i] > thr and sm[i] > sm[i-1] and sm[i] > sm[i+1]:
            peaks.append(i)
            last = i

    if len(peaks) < 2:
        return None

    intervals = [(peaks[i] - peaks[i-1]) / fs for i in range(1, len(peaks))]
    avg_period = sum(intervals) / len(intervals)
    if avg_period <= 0:
        return None
    return 60.0 / avg_period

def main():
    fs = 100
    m = MAX30102()
    m.setup(
        mode=MODE_SPO2,
        sample_avg=4,
        sample_rate=fs,
        pulse_width=411,
        adc_range=4096,
        led_red=0x30,
        led_ir=0x40
    )

    ir_raw_buf = deque(maxlen=fs * 10)   # 10초만 사용(더 빨리 반응)
    bpm_buf = deque(maxlen=5)
    last_print = time.time()
    last_sample = None

    # 손가락 감지 기준(당신 로그 기준으로 안전하게 잡음)
    # 손가락 올리면 IR이 150k~230k, 없으면 1k~10k 수준이었음
    FINGER_IR_MIN = 50000

    print("Reading MAX30102... Place finger steadily. Ctrl+C to stop.")
    try:
        while True:
            samples = m.read_samples(max_samples=32)
            if samples:
                last_sample = samples[-1]
                for red, ir in samples:
                    # 손가락 없으면 버퍼 리셋 (쓰레기 데이터 제거)
                    if ir < FINGER_IR_MIN:
                        ir_raw_buf.clear()
                        bpm_buf.clear()
                    else:
                        ir_raw_buf.append(ir)

            now = time.time()
            if now - last_print >= 1.0:
                last_print = now

                if last_sample:
                    red, ir = last_sample

                    if ir < FINGER_IR_MIN or len(ir_raw_buf) < fs * 4:
                        print(f"RED={red:6d} IR={ir:6d}  BPM=... (hold finger steady)")
                    else:
                        vals = list(ir_raw_buf)
                        mean = sum(vals) / len(vals)
                        sig = [v - mean for v in vals]

                        bpm = estimate_bpm_peaks(sig, fs)
                        if bpm is not None and 35 <= bpm <= 220:
                            bpm_buf.append(bpm)

                        bpm_smoothed = (sum(bpm_buf) / len(bpm_buf)) if bpm_buf else None

                        if bpm_smoothed is None:
                            print(f"RED={red:6d} IR={ir:6d}  BPM=...")
                        else:
                            print(f"RED={red:6d} IR={ir:6d}  BPM={bpm_smoothed:5.1f}")
                else:
                    print("No samples yet...")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExit")
    finally:
        m.close()

if __name__ == "__main__":
    main()
