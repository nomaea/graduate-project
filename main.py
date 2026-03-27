import argparse
import csv
import logging
import os
import socket
import struct
import threading
import time
from queue import Queue, Empty

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

import cv2
import numpy as np

from fer.fer_api import FERQueuePublisher
from fer.fer_core import FERCore
from fer.fer_visualizer import run_visualizer
from bio.bio_engine_core import main as bio_engine_main

from multimodal.real_fer_source import RealFerSource
from multimodal.real_sensor_source import RealSensorSource
from multimodal.multimodal_engine import MultiModalEngine
from multimodal.json_builder import fusion_result_to_json

# ▼▼▼ 분리한 ws_server.py 모듈에서 함수 불러오기 ▼▼▼
from multimodal.ws_server import start_ws_server, ws_broadcast


class SocketFrameReceiver:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None
        self.conn = None

    def open(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(1)
        print(f"[runtime] waiting camera service on {self.host}:{self.port}")
        self.conn, addr = self.server.accept()
        print(f"[runtime] connected by {addr}")

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed")
            buf += chunk
        return buf

    def read(self):
        header = self._recv_exact(4)
        (size,) = struct.unpack("!I", header)
        payload = self._recv_exact(size)
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame is not None, frame

    def release(self):
        try:
            if self.conn is not None:
                self.conn.close()
        finally:
            if self.server is not None:
                self.server.close()


class FERCSVLogger:
    def __init__(self):
        day = time.strftime("%Y%m%d")
        base = os.path.join(os.getcwd(), "logs", "fer", day)
        os.makedirs(base, exist_ok=True)
        self.path = os.path.join(base, f"fer_{time.strftime('%H%M%S')}.csv")
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "timestamp_unix",
            "packet_ts",
            "seq",
            "argmax_idx",
            "argmax_label",
            "face_detected",
            "latency_ms",
            "neutral",
            "happy",
            "sad",
            "angry",
        ])
        self.f.flush()
        print(f"[runtime] FER log -> {self.path}")

    def write(self, packet: dict):
        probs = packet.get("softmax", [0.0, 0.0, 0.0, 0.0])
        self.w.writerow([
            f"{time.time():.3f}",
            packet.get("ts", ""),
            packet.get("seq", ""),
            packet.get("argmax_idx", ""),
            packet.get("argmax_label", ""),
            int(bool(packet.get("face_detected", False))),
            packet.get("latency_ms", ""),
            probs[0] if len(probs) > 0 else "",
            probs[1] if len(probs) > 1 else "",
            probs[2] if len(probs) > 2 else "",
            probs[3] if len(probs) > 3 else "",
        ])
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class FusionCSVLogger:
    def __init__(self):
        day = time.strftime("%Y%m%d")
        base = os.path.join(os.getcwd(), "logs", "multimodal", day)
        os.makedirs(base, exist_ok=True)
        self.path = os.path.join(base, f"multimodal_{time.strftime('%H%M%S')}.csv")
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "timestamp_unix",
            "fusion_timestamp",
            "dominant_emotion",
            "confidence",
            "safe",
            "stressed",
            "alert_level",
            "alert_reason",
            "fer_neutral",
            "fer_happy",
            "fer_sad",
            "fer_angry",
            "fer_alert",
            "fer_drowsy",
            "bio_bpm",
            "bio_hrv",
            "bio_gsr",
            "bio_safe",
            "bio_stressed",
        ])
        self.f.flush()
        print(f"[runtime] Fusion log -> {self.path}")

    def write(self, fusion_result):
        fer_emo = fusion_result.fer.emotion_scores
        fer_drowsy = fusion_result.fer.drowsy_scores
        sensor_raw = fusion_result.sensor.raw_metrics
        sensor_emo = fusion_result.sensor.emotion_scores
        fused = fusion_result.fused_scores

        self.w.writerow([
            f"{time.time():.3f}",
            fusion_result.timestamp,
            fusion_result.dominant_emotion,
            fusion_result.confidence,
            fused.get("safe", 0.0),
            fused.get("stressed", 0.0),
            fusion_result.alert_level,
            fusion_result.alert_reason,
            fer_emo.get("neutral", 0.0),
            fer_emo.get("happy", 0.0),
            fer_emo.get("sad", 0.0),
            fer_emo.get("angry", 0.0),
            fer_drowsy.get("alert", 0.0),
            fer_drowsy.get("drowsy", 0.0),
            sensor_raw.get("bpm", 0.0),
            sensor_raw.get("hrv", 0.0),
            sensor_raw.get("gsr", 0.0),
            sensor_emo.get("safe", 0.0),
            sensor_emo.get("stressed", 0.0),
        ])
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class RuntimeController:
    def __init__(
        self,
        tflite_path: str,
        cam_index: int = 0,
        cam_w: int = 1280,
        cam_h: int = 720,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
        camera_backend: str = "v4l2",
        tcp_host: str = "127.0.0.1",
        tcp_port: int = 9999,
        flip_horizontal: bool = True,
    ):
        self.tflite_path = tflite_path
        self.cam_index = cam_index
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf
        self.camera_backend = camera_backend
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.flip_horizontal = flip_horizontal

        # 큐 사이즈 넉넉하게 수정 (미응답 병목 방지)
        self.fer_queue = Queue(maxsize=30)
        self.bio_payload_queue = Queue(maxsize=10)

        self.fer_core = None
        self.cap = None
        self.running = False

        self.fer_source = None
        self.sensor_source = None
        self.mm_engine = None
        self.bio_thread = None

        self.fer_logger = None
        self.fusion_logger = None

        self.last_fusion_result = None
        self.last_packet = None
        self._crop_window_open = False  # [FER#9] 창 생성 상태 추적

        # ▼▼▼ 분리한 ws_server 모듈을 스레드로 실행 ▼▼▼
        self.ws_thread = threading.Thread(target=start_ws_server, daemon=True)
        self.ws_thread.start()

    def _start_bio_thread(self):
        self.bio_thread = threading.Thread(
            target=bio_engine_main,
            kwargs={"payload_queue": self.bio_payload_queue},
            daemon=True,
        )
        self.bio_thread.start()
        print("[runtime] BIO thread started")

    def _drain_bio_queue(self):
        latest = None
        drained = 0
        try:
            while True:
                latest = self.bio_payload_queue.get_nowait()
                drained += 1
        except Empty:
            pass

        if latest is not None:
            self.sensor_source.update_window(latest)
            # [FER#10] 매 프레임 콘솔 출력 → logging.debug() 로 교체 (Raspberry Pi I/O 부하 방지)
            logging.debug(
                "[BIO_QUEUE] drained=%d bpm=%s bio_arousal=%s bio_conf=%s note=%s",
                drained, latest.get("bpm"), latest.get("bio_arousal"),
                latest.get("bio_conf"), latest.get("note"),
            )

    def init_components(self):
        fer_publisher = FERQueuePublisher(
            out_queue=self.fer_queue,
            debug_print=False,
        )

        self.fer_core = FERCore(
            tflite_path=self.tflite_path,
            publisher=fer_publisher,
            min_det_conf=self.min_det_conf,
            min_track_conf=self.min_track_conf,
        )

        self.fer_source = RealFerSource(self.fer_queue)
        self.sensor_source = RealSensorSource()
        self.mm_engine = MultiModalEngine(
            fer_source=self.fer_source,
            sensor_source=self.sensor_source,
            w_fer=0.4,
            w_sensor=0.6,
            ema_alpha=0.3,
        )

        self.fer_logger = FERCSVLogger()
        self.fusion_logger = FusionCSVLogger()

        self._start_bio_thread()

        if self.camera_backend == "tcp":
            self.cap = SocketFrameReceiver(self.tcp_host, self.tcp_port)
            self.cap.open()
        else:
            self.cap = cv2.VideoCapture(self.cam_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open webcam")

    def start_all(self):
        self.running = True

    def run_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break

            if self.flip_horizontal:
                frame = cv2.flip(frame, 1)

            fer_result = self.fer_core.process_frame(frame, return_debug=True)
            packet = fer_result["packet"]
            roi_preview = fer_result["roi_preview"]

            if packet is not None:
                self.last_packet = packet
                self.fer_logger.write(packet)

            self._drain_bio_queue()

            sensor_latest = self.sensor_source.get_latest_result()
            fer_latest = self.fer_source.get_latest_result()

            fusion_result = self.mm_engine.step()

            if fusion_result is not None:
                self.last_fusion_result = fusion_result
                self.fusion_logger.write(fusion_result)
                
                fusion_json_str = fusion_result_to_json(fusion_result)
                # [MUL#8] 매 프레임 JSON 전체 출력 → logging.debug()로 교체 (I/O 부하 방지)
                logging.debug("[FUSION_JSON] %s", fusion_json_str)

                ws_broadcast(fusion_json_str)

            display = frame.copy()
            cv2.putText(
                display,
                "Operation Mode: FER + BIO + MultiModal",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # FER 표시
            if self.last_packet is not None:
                probs = self.last_packet.get("softmax", [0, 0, 0, 0])
                label = self.last_packet.get("argmax_label", "N/A")
                cv2.putText(
                    display,
                    f"FER: {label}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"N={probs[0]:.2f} H={probs[1]:.2f} S={probs[2]:.2f} A={probs[3]:.2f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "FER: waiting...",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # BIO 표시
            if sensor_latest is not None:
                raw = sensor_latest.raw_metrics
                emo = sensor_latest.emotion_scores

                cv2.putText(
                    display,
                    f"BIO bpm={raw.get('bpm', 0):.1f} hrv={raw.get('hrv', 0):.4f} gsr={raw.get('gsr', 0):.1f}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"BIO safe={emo.get('safe', 0):.2f} stressed={emo.get('stressed', 0):.2f}",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "BIO: waiting...",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Fusion 표시
            if self.last_fusion_result is not None:
                fused = self.last_fusion_result.fused_scores
                cv2.putText(
                    display,
                    f"FUSION: {self.last_fusion_result.dominant_emotion}",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"CONF={self.last_fusion_result.confidence:.2f} SAFE={fused.get('safe', 0):.2f} STRESS={fused.get('stressed', 0):.2f}",
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"ALERT={self.last_fusion_result.alert_level}",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 0, 255) if self.last_fusion_result.alert_level != "NORMAL" else (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                reason = self.last_fusion_result.alert_reason or ""
                cv2.putText(
                    display,
                    reason[:80],
                    (20, 310),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (200, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    display,
                    "FUSION: waiting...",
                    (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 100, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                "q: quit",
                (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Operation Mode", display)

            if roi_preview is not None:
                roi_canvas = roi_preview.copy()
                cv2.putText(
                    roi_canvas,
                    "Aligned Face Crop",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Face Crop Preview", roi_canvas)
                self._crop_window_open = True
            else:
                # [FER#9] 창 상태 플래그로 관리 — getWindowProperty 불필요한 예외 방지
                if self._crop_window_open:
                    cv2.destroyWindow("Face Crop Preview")
                    self._crop_window_open = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False

    def stop_all(self):
        self.running = False

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        if self.fer_core is not None:
            self.fer_core.close()

        if self.fer_logger is not None:
            self.fer_logger.close()

        if self.fusion_logger is not None:
            self.fusion_logger.close()

        cv2.destroyAllWindows()


def run_operation_mode(
    tflite_path: str,
    cam_index: int,
    cam_w: int,
    cam_h: int,
    min_det_conf: float,
    min_track_conf: float,
    camera_backend: str,
    tcp_host: str,
    tcp_port: int,
    flip_horizontal: bool,
):
    controller = RuntimeController(
        tflite_path=tflite_path,
        cam_index=cam_index,
        cam_w=cam_w,
        cam_h=cam_h,
        min_det_conf=min_det_conf,
        min_track_conf=min_track_conf,
        camera_backend=camera_backend,
        tcp_host=tcp_host,
        tcp_port=tcp_port,
        flip_horizontal=flip_horizontal,
    )
    try:
        controller.init_components()
        controller.start_all()
        controller.run_loop()
    finally:
        controller.stop_all()


def main():
    print("[DEBUG] RUNNING MAIN FILE:", __file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "visualize"], default="run")
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--min_det_conf", type=float, default=0.5)
    parser.add_argument("--min_track_conf", type=float, default=0.5)
    parser.add_argument("--camera_backend", choices=["v4l2", "tcp"], default="v4l2")
    parser.add_argument("--tcp_host", default="127.0.0.1")
    parser.add_argument("--tcp_port", type=int, default=9999)
    parser.add_argument("--no_flip", action="store_true")
    args = parser.parse_args()

    if args.mode == "run":
        run_operation_mode(
            tflite_path=args.tflite,
            cam_index=args.cam_index,
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            min_det_conf=args.min_det_conf,
            min_track_conf=args.min_track_conf,
            camera_backend=args.camera_backend,
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            flip_horizontal=not args.no_flip,
        )
    else:
        run_visualizer(
            tflite_path=args.tflite,
            cam_index=args.cam_index,
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            min_det_conf=args.min_det_conf,
            min_track_conf=args.min_track_conf,
            camera_backend=args.camera_backend,
            tcp_host=args.tcp_host,
            tcp_port=args.tcp_port,
            flip_horizontal=not args.no_flip,
        )


if __name__ == "__main__":
    main()