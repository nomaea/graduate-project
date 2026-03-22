# main_ws.py
import asyncio
import json
from datetime import datetime, timezone, timedelta
from queue import Queue

import websockets

from fer_api import FERQueuePublisher
from fer_core import FERCore
from real_fer_source import RealFerSource
from real_sensor_source import RealSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json


# ── FER 큐 생성 및 FERCore 연결 ───────────────────────────
fer_queue = Queue(maxsize=1)
publisher = FERQueuePublisher(out_queue=fer_queue, debug_print=False)
fer_core  = FERCore(
    tflite_path="graduate-project-feature-fer/models/efficientface_4cls_finetuned_fp16_float16.tflite",
    publisher=publisher,
)

# ── 전역 초기화 ───────────────────────────────────────────
fer_src    = RealFerSource(fer_queue=fer_queue)
sensor_src = RealSensorSource()

engine = MultiModalEngine(
    fer_source=fer_src,
    sensor_source=sensor_src,
    w_fer=0.4,
    w_sensor=0.6,
    ema_alpha=0.3,
)

kst = timezone(timedelta(hours=9))

# ── 웹캠 전역 오픈 ────────────────────────────────────────
import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    print("[WARN] 웹캠을 열 수 없습니다.")
    cap = None


# ── 웹캠 루프 (백그라운드) ────────────────────────────────
async def camera_loop():
    """0.1초마다 웹캠 프레임 읽어서 FERCore 추론 → fer_queue에 자동 적재"""
    while True:
        if cap is not None:
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                fer_core.process_frame(frame)
        await asyncio.sleep(0.1)


# ── 앱으로 송신하는 루프 ──────────────────────────────────
async def send_loop(websocket):
    print("[INFO] 앱 클라이언트 연결됨")
    try:
        while True:
            result = engine.step()

            if result is not None:
                js = fusion_result_to_json(result)
                await websocket.send(js)

                data   = json.loads(js)
                fusion = data.get("fusion", {})
                alert  = data.get("alert", {})
                print(
                    f"[SEND] dominant={fusion.get('dominant')} "
                    f"safe={fusion.get('safe', 0):.3f} "
                    f"stressed={fusion.get('stressed', 0):.3f} "
                    f"alert={alert.get('level')}"
                )
            else:
                print("[INFO] 데이터 대기 중 (FER 또는 센서 없음)")

            await asyncio.sleep(1.0)

    except websockets.ConnectionClosed:
        print("[INFO] 앱 클라이언트 연결 끊김")


# ── 서버 시작 ─────────────────────────────────────────────
async def main():
    print("[INFO] 멀티모달 WebSocket 서버 시작")
    print("[INFO] 앱 접속 주소: ws://내노트북IP:8765")

    await asyncio.gather(
        camera_loop(),
        websockets.serve(send_loop, "0.0.0.0", 8765),
    )
    await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())