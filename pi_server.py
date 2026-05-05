from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
import subprocess
import threading
import time
import sys

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False
    print("[HTTP] psutil 없음 — CPU/메모리 모니터링 비활성화")

PROJECT_ROOT  = "/home/hyunwoo/Desktop/R_device"
MAIN_PYTHON   = sys.executable
CAMERA_PYTHON = "/usr/bin/python3"
CAMERA_HOST   = "127.0.0.1"
CAMERA_PORT   = 9999

main_process   = None
camera_process = None
lock = threading.Lock()

_monitor_thread = None
_monitor_stop   = threading.Event()
_cpu_main: list = []
_cpu_cam:  list = []
_mem_main: list = []
_mem_cam:  list = []
_psutil_procs: dict = {}  # pid -> psutil.Process (재사용으로 cpu_percent 정확도 확보)


def _monitor_fn():
    while not _monitor_stop.wait(timeout=1.0):
        with lock:
            main_pid = main_process.pid if main_process else None
            cam_pid  = camera_process.pid if camera_process else None

        if not _PSUTIL_OK:
            continue

        for pid, cpu_list, mem_list in [
            (main_pid, _cpu_main, _mem_main),
            (cam_pid,  _cpu_cam,  _mem_cam),
        ]:
            if pid is None:
                continue
            try:
                if pid not in _psutil_procs:
                    p = psutil.Process(pid)
                    p.cpu_percent(interval=None)  # 첫 호출은 항상 0.0 — 기준값 설정
                    _psutil_procs[pid] = p
                p = _psutil_procs[pid]
                cpu_list.append(p.cpu_percent(interval=None))
                mem_list.append(p.memory_percent())
            except psutil.NoSuchProcess:
                _psutil_procs.pop(pid, None)


def _avg(lst):
    return round(sum(lst) / len(lst), 2) if lst else 0.0


def _collect_metrics():
    result = {
        "main_avg_cpu_pct": _avg(_cpu_main),
        "main_avg_mem_pct": _avg(_mem_main),
        "cam_avg_cpu_pct":  _avg(_cpu_cam),
        "cam_avg_mem_pct":  _avg(_mem_cam),
    }
    # main.py 메트릭 (fps, latency)
    try:
        with open("/tmp/fer_main_metrics.json") as f:
            result.update(json.load(f))
    except Exception:
        pass
    # camera_service.py 메트릭 (카메라 fps)
    try:
        with open("/tmp/fer_camera_metrics.json") as f:
            data = json.load(f)
            result["camera_frame_count"] = data.get("frame_count")
            result["camera_avg_fps"]     = data.get("avg_fps")
    except Exception:
        pass
    return result


def start_engine():
    global main_process, camera_process, _monitor_thread

    with lock:
        if main_process is not None:
            print("[HTTP] 이미 실행 중")
            return

    # 이전 세션 메트릭 초기화
    _cpu_main.clear(); _cpu_cam.clear()
    _mem_main.clear(); _mem_cam.clear()
    _psutil_procs.clear()
    for path in ("/tmp/fer_main_metrics.json", "/tmp/fer_camera_metrics.json"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    main_cmd = [
        MAIN_PYTHON, "main.py",
        "--camera_backend", "tcp",
        "--tcp_host", CAMERA_HOST,
        "--tcp_port", str(CAMERA_PORT),
    ]
    with lock:
        main_process = subprocess.Popen(main_cmd, cwd=PROJECT_ROOT)
    print(f"[HTTP] 메인 엔진 시작 (PID: {main_process.pid})")

    time.sleep(5)

    camera_cmd = [
        CAMERA_PYTHON, "fer/camera_service.py",
        "--host", CAMERA_HOST,
        "--port", str(CAMERA_PORT),
        "--width", "640",
        "--height", "480",
        "--fps", "20",
    ]
    with lock:
        camera_process = subprocess.Popen(camera_cmd, cwd=PROJECT_ROOT)
    print(f"[HTTP] 카메라 서비스 시작 (PID: {camera_process.pid})")

    _monitor_stop.clear()
    _monitor_thread = threading.Thread(target=_monitor_fn, daemon=True)
    _monitor_thread.start()
    print("[HTTP] 모니터링 스레드 시작")


def stop_engine():
    global main_process, camera_process, _monitor_thread

    # 1. 모니터링 중단
    _monitor_stop.set()
    if _monitor_thread:
        _monitor_thread.join(timeout=3)
        _monitor_thread = None

    # 2. 프로세스 참조를 lock 안에서 가져온 뒤, lock 밖에서 종료 대기
    with lock:
        procs = [(main_process, "메인"), (camera_process, "카메라")]

    for proc, name in procs:
        if proc is None:
            continue
        try:
            proc.terminate()
            proc.wait(timeout=5)  # finally 블록에서 메트릭 파일 저장할 시간 확보
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"[HTTP] {name} 종료")

    with lock:
        main_process = None
        camera_process = None

    return _collect_metrics()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/start":
            threading.Thread(target=start_engine, daemon=True).start()
            self._respond(200, "OK")
        elif self.path == "/stop":
            metrics = stop_engine()
            self._respond_json(200, metrics)
        else:
            self._respond(404, "Not Found")

    def _respond(self, code, msg):
        self.send_response(code)
        self.end_headers()
        self.wfile.write(msg.encode())

    def _respond_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[HTTP] {args[0]} {args[1]}")


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 5000), Handler)
    print("[HTTP] Pi 서버 대기 중 (port=5000)")
    server.serve_forever()
