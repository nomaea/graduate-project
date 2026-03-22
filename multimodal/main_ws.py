# main_ws.py websocket코드
import asyncio
import json
from datetime import datetime, timezone, timedelta

import cv2
import websockets

from real_fer_source import RealFerSource
from real_sensor_source import RealSensorSource
from multimodal_engine import MultiModalEngine
from json_builder import fusion_result_to_json


# ---------- 1) 멀티모달 엔진 & 소스 초기화 (전역에서 한 번만) ----------
fer_src = RealFerSource()
sensor_src = RealSensorSource()

engine = MultiModalEngine(
    fer_source=fer_src,
    sensor_source=sensor_src,
    w_fer=0.4,
    w_sensor=0.6,
    ema_alpha=0.3,
)

kst = timezone(timedelta(hours=9))

# 센서 쪽도 main.py와 동일하게 샘플 payload 한 번 넣어둠 (동작 확인용)
sample_payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "user_id": "demo-user",
    "hr_series": [88, 92, 95, 97, 93],
    "eda_series": [0.18, 0.21, 0.20, 0.19],
    "acc_series": [0.03, 0.05, 0.04],
}
sensor_src.update_window(sample_payload)

# 웹캠 전역 오픈
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[WARN] 웹캠을 열 수 없습니다. FER는 항상 None일 수 있습니다.")
    cap = None


# ---------- 2) 클라이언트 하나당 실행되는 루프 ----------
async def send_loop(websocket):
    """
    WebSocket 클라이언트 한 명이 접속하면
    이 코루틴이 실행되면서 주기적으로 JSON을 보내줌.
    """
    print("[INFO] client connected")

    try:
        while True:
            # 1) FER 업데이트 (웹캠 프레임 기반)
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    fer_src.update_frame(frame)
                else:
                    print("[WARN] 프레임 읽기 실패 (FER 업데이트 생략)")

            # 2) 멀티모달 한 스텝 수행
            result = engine.step()

            if result is not None:
                # FusionResult -> JSON 문자열
                js = fusion_result_to_json(result)

                # 클라이언트에 전송
                await websocket.send(js)

                # 디버그 로그
                data = json.loads(js)
                fusion = data.get("fusion", {})
                safe = fusion.get("safe", 0.0)
                stressed = fusion.get("stressed", 0.0)
                dominant = fusion.get("dominant", "unknown")

                print(
                    f"[SEND] dominant={dominant} "
                    f"safe={safe:.3f} stressed={stressed:.3f}"
                )
            else:
                print("[INFO] 엔진 결과 없음 (FER/Sensor 데이터 부족)")

            # 전송 주기 (1초)
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
