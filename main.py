# main.py
import argparse
import json
import signal
import socket
import struct
import threading
import time
from datetime import datetime, timezone, timedelta
from queue import Empty, Queue

import cv2
import numpy as np

import sys
sys.path.insert(0, ".")

import bio.bio_engine_core as bio_engine_core
from fer.fer_api import FERQueuePublisher
from fer.fer_core import FERCore
from multimodal.json_builder import fusion_result_to_json
from multimodal.multimodal_engine import MultiModalEngine
from multimodal.real_fer_source import RealFerSource
from multimodal.real_sensor_source import RealSensorSource
from multimodal.ws_server import run_server_in_thread, ws_broadcast


def _recv_all(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return buf


def camera_tcp_thread(fer_core: FERCore, host: str, port: int, stop_event: threading.Event):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(1)
    server_sock.settimeout(1.0)
    print(f"[CAM] TCP 서버 대기 중 ({host}:{port})")

    while not stop_event.is_set():
        try:
            conn, addr = server_sock.accept()
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[CAM] accept 오류: {e}")
            break

        print(f"[CAM] camera_service 연결됨: {addr}")
        conn.settimeout(2.0)
        try:
            while not stop_event.is_set():
                header = _recv_all(conn, 4)
                length = struct.unpack("!I", header)[0]
                data   = _recv_all(conn, length)
                arr    = np.frombuffer(data, dtype=np.uint8)
                frame  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    fer_core.process_frame(frame)
        except Exception as e:
            print(f"[CAM] 연결 끊김: {e}")
        finally:
            conn.close()

    server_sock.close()
    print("[CAM] TCP 서버 종료")


def bio_thread_fn(payload_queue: Queue, stop_event: threading.Event):
    try:
        bio_engine_core.main(payload_queue=payload_queue)
    except Exception as e:
        print(f"[BIO] 스레드 종료: {e}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="run")
    parser.add_argument("--tflite",
                        default="fer/models/efficientface_4cls_finetuned_fp16_float16.tflite")
    parser.add_argument("--camera_backend", default="tcp")
    parser.add_argument("--tcp_host", default="127.0.0.1")
    parser.add_argument("--tcp_port", type=int, default=9999)
    parser.add_argument("--ws_port",  type=int, default=8765)
    args = parser.parse_args()

    def _sigterm(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _sigterm)

    session_start = time.time()
    stop_event = threading.Event()

    fer_queue = Queue(maxsize=1)
    publisher = FERQueuePublisher(out_queue=fer_queue, debug_print=False)
    fer_core  = FERCore(tflite_path=args.tflite, publisher=publisher)

    fer_src    = RealFerSource(fer_queue=fer_queue)
    sensor_src = RealSensorSource()

    engine = MultiModalEngine(
        fer_source=fer_src,
        sensor_source=sensor_src,
        w_fer=0.4,
        w_sensor=0.6,
        ema_alpha=0.3,
    )

    run_server_in_thread(host="0.0.0.0", port=args.ws_port)
    print(f"[WS] WebSocket 서버 기동 (port={args.ws_port})")

    if args.camera_backend == "tcp":
        cam_t = threading.Thread(
            target=camera_tcp_thread,
            args=(fer_core, args.tcp_host, args.tcp_port, stop_event),
            daemon=True,
        )
        cam_t.start()
        print(f"[CAM] 수신 스레드 기동 ({args.tcp_host}:{args.tcp_port})")

    bio_queue = Queue(maxsize=5)
    bio_t = threading.Thread(
        target=bio_thread_fn,
        args=(bio_queue, stop_event),
        daemon=True,
    )
    bio_t.start()
    print("[BIO] Bio 엔진 스레드 기동")

    print("[INFO] 멀티모달 엔진 시작\n")

    kst = timezone(timedelta(hours=9))

    try:
        while True:
            try:
                while True:
                    payload = bio_queue.get_nowait()
                    sensor_src.update_window(payload)
            except Empty:
                pass

            result = engine.step()

            if result is not None:
                js = fusion_result_to_json(result)
                ws_broadcast(js)

                data   = json.loads(js)
                ts     = data.get("timestamp")
                dt_str = (datetime.fromtimestamp(ts, tz=timezone.utc)
                          .astimezone(kst).strftime("%H:%M:%S")) if ts else "??:??:??"
                fusion = data.get("fusion", {})
                alert  = data.get("alert", {})
                print(
                    f"[{dt_str}] "
                    f"alert={alert.get('level')} "
                    f"dominant={fusion.get('dominant')} "
                    f"conf={fusion.get('confidence', 0):.2f}"
                )
            else:
                print("[INFO] 데이터 대기 중...")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] 종료")
    finally:
        stop_event.set()
        fer_core.close()

        session_duration = time.time() - session_start
        m = fer_core.get_metrics()
        fps = m["fer_frame_count"] / session_duration if session_duration > 0 else 0.0
        metrics = {
            "session_duration_s": round(session_duration, 2),
            "fer_frame_count":    m["fer_frame_count"],
            "avg_fps":            round(fps, 2),
            "avg_latency_ms":     m["avg_latency_ms"],
            "min_latency_ms":     m["min_latency_ms"],
            "max_latency_ms":     m["max_latency_ms"],
        }
        try:
            with open("/tmp/fer_main_metrics.json", "w") as f:
                json.dump(metrics, f)
            print(f"[INFO] 세션 메트릭 저장 완료: {metrics}")
        except Exception as e:
            print(f"[INFO] 메트릭 저장 실패: {e}")


if __name__ == "__main__":
    main()
