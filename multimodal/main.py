# main.py
import time
import json
from datetime import datetime, timezone, timedelta
from queue import Queue

from real_fer_source import RealFerSource
from real_sensor_source import RealSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json
from shared_queue import fer_queue


def main():
    # ── 소스 및 엔진 초기화 ───────────────────────────────
    # fer_queue는 shared_queue.py에서 가져옴
    # FER 친구가 자기 코드에서 fer_queue에 넣어줌
    fer_src    = RealFerSource(fer_queue=fer_queue)
    sensor_src = RealSensorSource()

    engine = MultiModalEngine(
        fer_source=fer_src,
        sensor_source=sensor_src,
        w_fer=0.4,
        w_sensor=0.6,
        ema_alpha=0.3,
    )

    print("[INFO] 멀티모달 엔진 시작")
    print("[INFO] FER 친구 데이터 대기 중...")
    print("[INFO] 찬희 BIO 데이터 대기 중...\n")

    kst      = timezone(timedelta(hours=9))
    step_idx = 0

    try:
        while True:
            result   = engine.step()
            step_idx += 1
            print(f"\n========== STEP {step_idx} ==========")

            if result is None:
                print("  [INFO] FER 또는 센서 데이터 대기 중...\n")
            else:
                js   = fusion_result_to_json(result)
                data = json.loads(js)

                # 타임스탬프
                ts = data.get("timestamp")
                if ts:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(kst)
                    print(f"[Timestamp] {dt.strftime('%Y-%m-%d %H:%M:%S')} (KST)")

                # Fusion
                fusion = data.get("fusion", {})
                print(f"\n[Fusion]")
                print(f"  safe      : {fusion.get('safe', 0):.3f}")
                print(f"  stressed  : {fusion.get('stressed', 0):.3f}")
                print(f"  dominant  : {fusion.get('dominant')}")
                print(f"  confidence: {fusion.get('confidence', 0):.3f}")

                # Alert
                alert = data.get("alert", {})
                print(f"\n[Alert]")
                print(f"  level : {alert.get('level')}")
                print(f"  reason: {alert.get('reason')}")

                # Sensor
                raw = data.get("sensor", {}).get("raw_metrics", {})
                print(f"\n[Sensor]")
                print(f"  bpm: {raw.get('bpm', 0):.1f}")
                print(f"  hrv: {raw.get('hrv', 0):.3f}")
                print(f"  gsr: {raw.get('gsr', 0):.1f}")

                # FER
                emo    = data.get("fer", {}).get("emotion_scores", {})
                drowsy = data.get("fer", {}).get("drowsy_scores", {})
                print(f"\n[FER]")
                print(f"  neutral: {emo.get('neutral', 0):.3f}")
                print(f"  happy  : {emo.get('happy', 0):.3f}")
                print(f"  sad    : {emo.get('sad', 0):.3f}")
                print(f"  angry  : {emo.get('angry', 0):.3f}")
                print(f"  alert  : {drowsy.get('alert', 0):.3f}")
                print(f"  drowsy : {drowsy.get('drowsy', 0):.3f}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] 종료")


if __name__ == "__main__":
    main()