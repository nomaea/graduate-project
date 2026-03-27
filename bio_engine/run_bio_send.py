import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from multimodal.real_sensor_source import RealSensorSource
import bio_engine_v2p5_realtime as bio_engine


sensor_src = real_sensor_source()


def handle_bio_payload(payload: dict) -> None:
    send_payload = {
        "timestamp": payload["timestamp"],
        "user_id": "bio-engine",
        "ppg_hr": payload["bpm"] if payload["bpm"] is not None else 0.0,
        "gsr_val": payload["gsr_cur"] if payload["gsr_cur"] is not None else 0.0,
    }
    sensor_src.update_window(send_payload)
    print("[BIO -> MULTIMODAL]", send_payload)


if __name__ == "__main__":
    bio_engine.main(on_payload=handle_bio_payload)
