# main.py
import time
import json
from datetime import datetime, timezone, timedelta

from mock.fer_mock import MockFerSource
from mock.sensor_mock import MockSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json


def main() -> None:
    # Mock 소스 초기화 (나중에 RealFerSource / RealSensorSource로 교체 가능)
    fer_src = MockFerSource()
    sensor_src = MockSensorSource()

    # 멀티모달 엔진 초기화
    engine = MultiModalEngine(
        fer_source=fer_src,
        sensor_source=sensor_src,
        w_fer=0.4,      # FER 가중치
        w_sensor=0.6,   # 센서 가중치
        ema_alpha=0.3,  # EMA 계수
    )

    print("[INFO] Multimodal engine started (mock mode)\n")

    kst = timezone(timedelta(hours=9))

    for step_idx in range(3):
        result = engine.step()

        print(f"\n========== STEP {step_idx + 1} ==========")

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


if __name__ == "__main__":
    main()
