import os
import sys
import time
import threading
from queue import Queue, Empty

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import bio_engine_core
from multimodal.real_sensor_source import RealSensorSource


def fmt_num(value, digits=2, empty="..."):
    if value is None:
        return empty
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return empty


def fmt_int_bool(value):
    return "1" if value else "0"


def short_note(note: str) -> str:
    if not note:
        return "-"
    if note == "hrv_warming_up(need~30s)":
        return "hrv_warming_up"
    return note


def print_bio_payload(payload: dict) -> None:
    ts = str(payload.get("timestamp", ""))
    ts_short = ts[-8:] if len(ts) >= 8 else ts

    print(
        f"[BIO] ts={ts_short} "
        f"arousal={fmt_num(payload.get('bio_arousal'))} "
        f"conf={fmt_num(payload.get('bio_conf'))} "
        f"bpm={fmt_num(payload.get('bpm'), 1)} "
        f"rmssd={fmt_num(payload.get('rmssd_use'), 3)} "
        f"gsr={fmt_num(payload.get('gsr_cur'), 1)} "
        f"ppgQ={fmt_num(payload.get('ppgQ'))} "
        f"finger={fmt_int_bool(payload.get('finger_on', False))} "
        f"gsr_fresh={fmt_int_bool(payload.get('gsr_fresh', False))} "
        f"note={short_note(str(payload.get('note', '')))}"
    )


def print_multimodal_result(latest_result) -> None:
    raw = latest_result.raw_metrics
    emo = latest_result.emotion_scores

    print(
        f"[MM ] "
        f"bpm={fmt_num(raw.get('bpm'), 1)} "
        f"hrv={fmt_num(raw.get('hrv'), 3)} "
        f"gsr={fmt_num(raw.get('gsr'), 1)} "
        f"safe={fmt_num(emo.get('safe'))} "
        f"stressed={fmt_num(emo.get('stressed'))}"
    )


def main():
    payload_queue = Queue(maxsize=5)
    sensor_src = RealSensorSource()

    bio_thread = threading.Thread(
        target=bio_engine_core.main,
        kwargs={"payload_queue": payload_queue},
        daemon=True,
    )
    bio_thread.start()

    last_timestamp = None

    while True:
        latest_payload = None

        try:
            while True:
                latest_payload = payload_queue.get_nowait()
        except Empty:
            pass

        if latest_payload is not None:
            current_timestamp = latest_payload.get("timestamp")

            # 같은 timestamp 중복 출력 방지
            if current_timestamp != last_timestamp:
                sensor_src.update_window(latest_payload)

                print_bio_payload(latest_payload)

                latest_result = sensor_src.get_latest_result()
                if latest_result is not None:
                    print_multimodal_result(latest_result)

                print("-" * 90)
                last_timestamp = current_timestamp

        time.sleep(0.2)


if __name__ == "__main__":
    main()
