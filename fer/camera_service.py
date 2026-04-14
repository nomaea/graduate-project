# -*- coding: utf-8 -*-
import argparse
import socket
import struct
import time

import cv2
from picamera2 import Picamera2


def send_all(sock: socket.socket, data: bytes):
    view = memoryview(data)
    while len(view):
        sent = sock.send(view)
        if sent == 0:
            raise ConnectionError("Socket connection broken")
        view = view[sent:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    args = parser.parse_args()

    print("[camera_service] init Picamera2")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)

    print(f"[camera_service] connect {args.host}:{args.port}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print("[camera_service] connected")

    frame_interval = 1.0 / max(args.fps, 1)
    last_time = 0.0
    frame_count = 0

    try:
        while True:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.001)
                continue
            last_time = now

            # Picamera2 프레임 획득
            frame = picam2.capture_array()
            if frame is None:
                print("[camera_service] frame capture failed")
                continue

            # 필요 시 디버그 저장
            # if frame_count == 0:
            #     cv2.imwrite("debug_camera_frame.jpg", frame)

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            if not ok:
                print("[camera_service] jpeg encode failed")
                continue

            payload = encoded.tobytes()
            header = struct.pack("!I", len(payload))
            send_all(sock, header)
            send_all(sock, payload)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[camera_service] sent {frame_count} frames")

    except KeyboardInterrupt:
        print("[camera_service] stopped by user")
    finally:
        try:
            sock.close()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
