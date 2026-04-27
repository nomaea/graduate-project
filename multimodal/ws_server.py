"""
ws_server.py — WebSocket broadcast server for multimodal fusion results.

Threading model
───────────────
  Main thread (camera loop)
    └─ ws_broadcast()
         └─ loop.call_soon_threadsafe(_enqueue)   ← 유일한 크로스-스레드 진입점

  WS thread (asyncio event loop)
    ├─ _ws_handler()          — 클라이언트 연결/해제, _ws_clients 관리
    └─ _broadcast_worker()    — _msg_queue 소비 → 모든 클라이언트에 전송

_ws_clients 는 오직 asyncio event loop 안에서만 변경된다.
크로스-스레드 공유 없음 → Lock 불필요.
"""
import asyncio
import logging
import threading

import websockets
try:
    from websockets.exceptions import WebSocketException
except ImportError:                      # websockets 구버전 호환
    WebSocketException = Exception       # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# ── 설정 상수 ──────────────────────────────────────────────────────────────────
_DEFAULT_HOST  = "0.0.0.0"
_DEFAULT_PORT  = 8765
_PING_INTERVAL = 20          # 초: 연결 유지 ping 주기
_PING_TIMEOUT  = 20          # 초: ping 무응답 허용 시간
_QUEUE_MAXSIZE = 64          # 백프레셔: 최대 미전송 메시지 수
_STOP_SENTINEL = object()    # _broadcast_worker 정상 종료 신호

# ── asyncio loop 전용 상태 (외부 스레드에서 직접 접근 금지) ───────────────────
_ws_clients: set = set()
_msg_queue: "asyncio.Queue | None" = None
_shutdown_event: "asyncio.Event | None" = None

# ── 스레드 간 공유 (쓰기는 WS thread 기동 시 1회, 이후 읽기 전용) ─────────────
_ws_loop: "asyncio.AbstractEventLoop | None" = None
_server_ready = threading.Event()   # 서버 listen 완료 신호 (main thread 대기용)


# ── 클라이언트 핸들러 ──────────────────────────────────────────────────────────

async def _ws_handler(websocket):
    """새 클라이언트 연결 시 호출. 연결 해제까지 블록."""
    _ws_clients.add(websocket)
    logger.info("[WS] 클라이언트 연결 (현재: %d)", len(_ws_clients))
    try:
        await websocket.wait_closed()
    except WebSocketException as e:
        logger.warning("[WS] 비정상 연결 종료: %s", e)
    finally:
        _ws_clients.discard(websocket)   # add 미완료 시에도 안전하게 제거
        logger.info("[WS] 클라이언트 해제 (현재: %d)", len(_ws_clients))


# ── 브로드캐스트 워커 ──────────────────────────────────────────────────────────

async def _broadcast_worker():
    """
    _msg_queue 를 소비해 연결된 모든 클라이언트에 동시 전송.
    _STOP_SENTINEL 수신 시 정상 종료.
    """
    while True:
        msg = await _msg_queue.get()

        if msg is _STOP_SENTINEL:
            break

        if not _ws_clients:
            continue

        # ① 스냅샷 고정
        #    tasks 생성과 결과 매핑을 동일한 리스트로 수행 → zip 인덱스 정합 보장
        #    await 사이에 _ws_clients 가 변경되어도 영향 없음
        clients = list(_ws_clients)
        tasks   = [asyncio.create_task(c.send(msg)) for c in clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # ② 동일 스냅샷 기준으로 에러 클라이언트 제거
        for client, result in zip(clients, results):
            if isinstance(result, Exception):
                _ws_clients.discard(client)
                logger.debug(
                    "[WS] 좀비 클라이언트 정리: %s", type(result).__name__
                )


# ── 서버 코루틴 ────────────────────────────────────────────────────────────────

async def _ws_serve(host: str, port: int):
    global _msg_queue, _shutdown_event

    _shutdown_event = asyncio.Event()
    _msg_queue      = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)

    asyncio.create_task(_broadcast_worker())

    try:
        async with websockets.serve(
            _ws_handler, host, port,
            ping_interval=_PING_INTERVAL,
            ping_timeout=_PING_TIMEOUT,
        ):
            logger.info("[WS] WebSocket 서버 대기 중 (ws://%s:%d)", host, port)
            _server_ready.set()              # 메인 스레드 unblock
            await _shutdown_event.wait()     # stop_ws_server() 호출까지 블록

    except OSError as e:
        logger.error("[WS] 서버 기동 실패 — 포트 %d 충돌 또는 권한 없음: %s", port, e)
    finally:
        _server_ready.set()                  # 실패 시에도 대기 스레드 unblock


# ── 스레드 진입점 ──────────────────────────────────────────────────────────────

def start_ws_server(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT):
    """WS 스레드 내부 진입점. 직접 호출 대신 run_server_in_thread() 를 사용."""
    global _ws_loop
    _ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_ws_loop)
    _ws_loop.run_until_complete(_ws_serve(host, port))


# ── 공개 API ───────────────────────────────────────────────────────────────────

def run_server_in_thread(
    host: str = _DEFAULT_HOST,
    port: int = _DEFAULT_PORT,
) -> threading.Thread:
    """
    WS 서버를 데몬 스레드로 기동.
    서버가 실제로 listen 상태가 된 후 반환 (최대 5초 대기).
    """
    t = threading.Thread(
        target=start_ws_server,
        kwargs={"host": host, "port": port},
        daemon=True,
        name="ws-server",
    )
    t.start()
    ready = _server_ready.wait(timeout=5.0)
    if not ready:
        logger.error("[WS] 서버가 5초 내 준비 완료되지 않음")
    return t


def stop_ws_server():
    """
    Graceful shutdown.
    _broadcast_worker 를 정지시키고, websockets.serve 컨텍스트를 종료해
    연결된 클라이언트에게 WebSocket close frame 을 전송한다.
    """
    if _ws_loop is None or not _ws_loop.is_running():
        return

    def _trigger():
        if _shutdown_event is not None:
            _shutdown_event.set()
        # STOP_SENTINEL 을 큐에 삽입해 _broadcast_worker 를 깨워 종료
        if _msg_queue is not None:
            try:
                _msg_queue.put_nowait(_STOP_SENTINEL)
            except asyncio.QueueFull:
                pass   # 워커가 곧 _shutdown_event 를 통해 종료됨

    _ws_loop.call_soon_threadsafe(_trigger)


def ws_broadcast(fusion_json: str):
    """
    메인 스레드에서 호출.
    fusion JSON 문자열을 asyncio 큐에 스레드 안전하게 삽입한다.

    백프레셔 정책: 큐가 꽉 찼을 때 가장 오래된 메시지를 드롭하고 신규 삽입.
    (최신 추론 결과 우선 전달)
    """
    # 서버 준비 전, 또는 루프가 종료된 경우 조기 반환
    if not _server_ready.is_set():
        return
    if _ws_loop is None or not _ws_loop.is_running():
        return

    def _enqueue():
        if _msg_queue is None:
            return
        if _msg_queue.full():
            # 가장 오래된 메시지 드롭 후 삽입 (frame-drop 정책)
            try:
                _msg_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            _msg_queue.put_nowait(fusion_json)
        except asyncio.QueueFull:
            pass   # 극단적 경합 — 다음 프레임에서 재시도

    # call_soon_threadsafe: 크로스-스레드 유일 진입점, GIL-free 안전
    _ws_loop.call_soon_threadsafe(_enqueue)
