# main.py
import time
import json
from datetime import datetime, timezone, timedelta

import cv2

from real_fer_source import RealFerSource
from real_sensor_source import RealSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json


def main() -> None:
    # 1) 실제 소스 초기화
    fer_src = RealFerSource()        # TFLite FER + drowsy
    sensor_src = RealSensorSource()  # BIO 파이프라인 (BioService 기반)

    # 2) 멀티모달 엔진 초기화
    engine = MultiModalEngine(
        fer_source=fer_src,
        sensor_source=sensor_src,
        w_fer=0.4,      # FER 가중치
        w_sensor=0.6,   # 센서 가중치
        ema_alpha=0.3,  # EMA 계수
    )

    print("[INFO] Multimodal engine started (REAL FER + REAL BIO mock payload)\n")

    # 3) 센서 쪽은 일단 샘플 payload 한 번 넣어서 동작 확인
    sample_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": "demo-user",
        "hr_series": [88, 92, 95, 97, 93],
        "eda_series": [0.18, 0.21, 0.20, 0.19],
        "acc_series": [0.03, 0.05, 0.04],
    }
    sensor_src.update_window(sample_payload)

    # 4) 웹캠 오픈
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    kst = timezone(timedelta(hours=9))

    step_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] 프레임 읽기 실패")
                break

            # 5) FER 업데이트 (현재 프레임 기반)
            fer_src.update_frame(frame)

            # 6) 멀티모달 한 스텝 수행
            result = engine.step()

            step_idx += 1
            print(f"\n========== STEP {step_idx} ==========")

            if result is None:
                print("  [INFO] FER 혹은 Sensor 데이터가 아직 없습니다.\n")
            else:
                # 1) FusionResult -> JSON 문자열
                js = fusion_result_to_json(result)

                print("[RAW JSON]")
                print(js)

                # 2) JSON 문자열 -> dict 로 다시 파싱해서 사람이 보기 좋게 출력
                data = json.loads(js)

                # timestamp
                ts = data.get("timestamp")
                print("\n[Timestamp]")
                if ts is not None:
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(kst)
                    print(f"  {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (KST)")
                else:
                    print("  <no timestamp>")

                # Fusion
                fusion = data.get("fusion", {})
                print("\n[Fusion 결과]")
                print(f"  safe      : {fusion.get('safe', 0.0):.3f}")
                print(f"  stressed  : {fusion.get('stressed', 0.0):.3f}")
                print(f"  dominant  : {fusion.get('dominant', 'unknown')}")
                print(f"  confidence: {fusion.get('confidence', 0.0):.3f}")
                print()

                # Alert
                alert = data.get("alert", {})
                print("[Alert]")
                print(f"  level : {alert.get('level', 'UNKNOWN')}")
                print(f"  reason: {alert.get('reason', '')}")
                print()

                # Sensor
                sensor = data.get("sensor", {})
                raw = sensor.get("raw_metrics", {})
                print("[Sensor(raw)]")
                print(f"  hrv : {raw.get('hrv', 0.0):.3f}")
                print(f"  gsr : {raw.get('gsr', 0.0):.3f}")
                print()

                # FER
                fer = data.get("fer", {})
                emo = fer.get("emotion_scores", {})
                drowsy = fer.get("drowsy_scores", {})

                print("[FER]")
                print("  emotion_scores:")
                print(f"    angry  : {emo.get('angry', 0.0):.3f}")
                print(f"    sad    : {emo.get('sad', 0.0):.3f}")
                print(f"    happy  : {emo.get('happy', 0.0):.3f}")
                print(f"    neutral: {emo.get('neutral', 0.0):.3f}")
                print("  drowsy_scores:")
                print(f"    alert  : {drowsy.get('alert', 0.0):.3f}")
                print(f"    drowsy : {drowsy.get('drowsy', 0.0):.3f}")
                print()

            # STEP 간 간격 (1초)
            time.sleep(1.0)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
