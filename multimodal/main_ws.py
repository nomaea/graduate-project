# main_ws.py
import asyncio
import json
from datetime import datetime, timezone, timedelta

import websockets

from real_fer_source import RealFerSource
from real_sensor_source import RealSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json
from shared_queue import fer_queue


# ── 전역 초기화 ───────────────────────────────────────────
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

kst = timezone(timedelta(hours=9))


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

    async with websockets.serve(send_loop, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())