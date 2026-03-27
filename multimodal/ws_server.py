# ws_server.py
import asyncio
import websockets

_ws_clients = set()
_ws_loop = None

async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    print(f"[WS] 앱 클라이언트 연결됨! (현재 접속: {len(_ws_clients)}대)")
    try:
        await websocket.wait_closed()
    finally:
        _ws_clients.remove(websocket)
        print(f"[WS] 앱 클라이언트 연결 해제. (현재 접속: {len(_ws_clients)}대)")

async def _ws_serve():
    async with websockets.serve(_ws_handler, "0.0.0.0", 8765):
        await asyncio.Future()

def start_ws_server():
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    print("[WS] WebSocket 서버 백그라운드 시작 (ws://0.0.0.0:8765)")
    _ws_loop.run_until_complete(_ws_serve())

def ws_broadcast(fusion_json: str):
    if not _ws_clients or _ws_loop is None:
        return
    async def send_to_all():
        if _ws_clients:
            await asyncio.gather(*[client.send(fusion_json) for client in _ws_clients], return_exceptions=True)
    asyncio.run_coroutine_threadsafe(send_to_all(), _ws_loop)