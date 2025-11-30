# main_ws.py websocket코드
import asyncio
import json
from datetime import datetime, timezone, timedelta

import websockets

from mock.fer_mock import MockFerSource
from mock.sensor_mock import MockSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json


# ---------- 1) 멀티모달 엔진 초기화 (전역에서 한 번만) ----------
fer_src = MockFerSource()
sensor_src = MockSensorSource()

engine = MultiModalEngine(
    fer_source=fer_src,
    sensor_source=sensor_src,
    w_fer=0.4,
    w_sensor=0.6,
    ema_alpha=0.3,
)

kst = timezone(timedelta(hours=9))


# ---------- 2) 클라이언트 하나당 실행되는 루프 ----------
async def send_loop(websocket):
    """
    WebSocket 클라이언트 한 명이 접속하면
    이 코루틴이 실행되면서 주기적으로 JSON을 보내줌.
    """
    print("[INFO] client connected")

    try:
        while True:
            # 멀티모달 한 스텝 수행
            result = engine.step()

            if result is not None:
                # FusionResult -> JSON 문자열
                js = fusion_result_to_json(result)

                # WebSocket으로 JSON 전송
                await websocket.send(js)

                # 서버 로그용 보기 좋은 출력
                data = json.loads(js)
                ts = data.get("timestamp")
                dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(kst)

                fusion = data.get("fusion", {})
                safe = fusion.get("safe", 0.0)
                stressed = fusion.get("stressed", 0.0)
                dominant = fusion.get("dominant", "unknown")

                print(
                    f"[SEND] {dt.strftime('%H:%M:%S')} "
                    f"dominant={dominant} safe={safe:.3f} stressed={stressed:.3f}"
                )

            # 1초마다 한 번씩 전송 (주기 조절 가능)
            await asyncio.sleep(1.0)

    except websockets.ConnectionClosed:
        print("[INFO] client disconnected")


# ---------- 3) WebSocket 서버 시작 ----------
async def main():
    """
    WebSocket 서버 열기
      - 호스트: 0.0.0.0 (모든 IP에서 접근 가능)
      - 포트  : 8765
      - 주소  : ws://<이 노트북 IP>:8765
    """
    host = "0.0.0.0"
    port = 8765

    print(f"[INFO] Multimodal WebSocket server starting on ws://{host}:{port}")

    async with websockets.serve(send_loop, host, port):
        print("[INFO] WebSocket server started. Waiting for clients...")
        # 서버를 계속 살아있게 유지
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
