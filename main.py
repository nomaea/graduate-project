import argparse
import time
import cv2
from queue import Queue, Empty

from fer_api import FERQueuePublisher
from fer_core import FERCore
from fer_visualizer import run_visualizer


class BIOEngineStub:
    def __init__(self, out_queue: Queue):
        self.out_queue = out_queue
        self.running = False
        self.seq = 0

    def start(self):
        self.running = True

    def step(self):
        if not self.running:
            return

        packet = {
            "ts": time.monotonic(),
            "seq": self.seq,
            "class_order": ["neutral", "happy", "sad", "angry"],
            "softmax": [0.25, 0.25, 0.25, 0.25],
            "source": "bio",
        }
        self.seq += 1

        try:
            self.out_queue.put_nowait(packet)
        except Exception:
            try:
                self.out_queue.get_nowait()
            except Empty:
                pass
            self.out_queue.put_nowait(packet)

    def stop(self):
        self.running = False


class FusionEngineStub:
    def __init__(self, fer_queue: Queue, bio_queue: Queue):
        self.fer_queue = fer_queue
        self.bio_queue = bio_queue
        self.running = False
        self.latest_fer = None
        self.latest_bio = None

    def start(self):
        self.running = True

    def update_inputs(self):
        if not self.running:
            return

        try:
            self.latest_fer = self.fer_queue.get_nowait()
        except Empty:
            pass

        try:
            self.latest_bio = self.bio_queue.get_nowait()
        except Empty:
            pass

    def step(self):
        if not self.running:
            return

        self.update_inputs()

        if self.latest_fer is None or self.latest_bio is None:
            return

        print("[Fusion] FER =", self.latest_fer["softmax"])
        print("[Fusion] BIO =", self.latest_bio["softmax"])

    def stop(self):
        self.running = False


class RuntimeController:
    def __init__(
        self,
        tflite_path: str,
        cam_index: int = 0,
        cam_w: int = 1280,
        cam_h: int = 720,
        min_det_conf: float = 0.5,
        min_track_conf: float = 0.5,
    ):
        self.tflite_path = tflite_path
        self.cam_index = cam_index
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf

        self.fer_queue = Queue(maxsize=1)
        self.bio_queue = Queue(maxsize=1)

        self.fer_core = None
        self.bio_engine = None
        self.fusion_engine = None
        self.cap = None
        self.running = False

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

        self.bio_engine = BIOEngineStub(out_queue=self.bio_queue)
        self.fusion_engine = FusionEngineStub(
            fer_queue=self.fer_queue,
            bio_queue=self.bio_queue,
        )

        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_h)

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam")

    def start_all(self):
        self.bio_engine.start()
        self.fusion_engine.start()
        self.running = True

    def run_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # FER 추론 + debug info
            fer_result = self.fer_core.process_frame(frame, return_debug=True)
            packet = fer_result["packet"]
            roi_preview = fer_result["roi_preview"]

            # 운영 모드 기본 화면
            display = frame.copy()
            cv2.putText(
                display,
                "Operation Mode: FER + BIO + Fusion",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if packet is not None:
                label = packet["argmax_label"]
                probs = packet["softmax"]

                cv2.putText(
                    display,
                    f"Emotion: {label}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"N={probs[0]:.2f} H={probs[1]:.2f} S={probs[2]:.2f} A={probs[3]:.2f}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                display,
                "q: quit",
                (20, self.cam_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Operation Mode", display)

            # 얼굴 crop preview
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
            else:
                try:
                    if cv2.getWindowProperty("Face Crop Preview", cv2.WND_PROP_VISIBLE) >= 0:
                        cv2.destroyWindow("Face Crop Preview")
                except cv2.error:
                    pass

            # BIO / Fusion
            self.bio_engine.step()
            self.fusion_engine.step()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False

    def stop_all(self):
        self.running = False

        if self.bio_engine is not None:
            self.bio_engine.stop()

        if self.fusion_engine is not None:
            self.fusion_engine.stop()

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()

        if self.fer_core is not None:
            self.fer_core.close()


def run_operation_mode(
    tflite_path: str,
    cam_index: int,
    cam_w: int,
    cam_h: int,
    min_det_conf: float,
    min_track_conf: float,
):
    controller = RuntimeController(
        tflite_path=tflite_path,
        cam_index=cam_index,
        cam_w=cam_w,
        cam_h=cam_h,
        min_det_conf=min_det_conf,
        min_track_conf=min_track_conf,
    )
    try:
        controller.init_components()
        controller.start_all()
        controller.run_loop()
    finally:
        controller.stop_all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "visualize"], default="run")
    parser.add_argument("--tflite", required=True)
    parser.add_argument("--cam_index", type=int, default=0)
    parser.add_argument("--cam_w", type=int, default=1280)
    parser.add_argument("--cam_h", type=int, default=720)
    parser.add_argument("--min_det_conf", type=float, default=0.5)
    parser.add_argument("--min_track_conf", type=float, default=0.5)
    args = parser.parse_args()

    if args.mode == "run":
        run_operation_mode(
            tflite_path=args.tflite,
            cam_index=args.cam_index,
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            min_det_conf=args.min_det_conf,
            min_track_conf=args.min_track_conf,
        )
    elif args.mode == "visualize":
        run_visualizer(
            tflite_path=args.tflite,
            cam_index=args.cam_index,
            cam_w=args.cam_w,
            cam_h=args.cam_h,
            min_det_conf=args.min_det_conf,
            min_track_conf=args.min_track_conf,
        )


if __name__ == "__main__":
    main()