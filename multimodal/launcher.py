import subprocess
import time
import threading
import serial
import os

# ─────────────────────────────────────────
#절대 경로 자동 추적 로직
# ─────────────────────────────────────────
# 1. 현재 launcher.py가 있는 폴더 찾기
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 2. 부모 폴더를 최상위 Root로 지정 
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# ─────────────────────────────────────────
# 설정값 (환경에 맞게 수정)
# ─────────────────────────────────────────
SERIAL_PORT   = "/dev/ttyUSB0"   # Windows면 "COM3" 등으로 변경
SERIAL_BAUD   = 9600

CAMERA_HOST   = "127.0.0.1"
CAMERA_PORT   = 9999

# 실제 시리얼 패킷 문자열 (앱과 맞춰서 수정)
PACKET_START  = "START"
PACKET_STOP   = "STOP"

# 가상환경 및 시스템 파이썬 설정
CAMERA_PYTHON = "/usr/bin/python3" # 라즈베리파이 파이썬
MAIN_PYTHON   = "python"  # 활성화된 가상환경 파이썬

class SystemLauncher:
    def __init__(self):
        self.main_process   = None
        self.camera_process = None
        self._lock          = threading.Lock()  # 동시 실행/종료 방지

        print("[System] 런처 초기화 시작...")

        # 시리얼 포트 초기화
        try:
            self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            print(f"[System] 시리얼 포트 연결 완료: {SERIAL_PORT} @ {SERIAL_BAUD}bps")
        except Exception as e:
            print(f"[System] 시리얼 포트 연결 실패: {e}")
            print("[System] 테스트 모드(키보드 입력)로 전환합니다.")
            self.ser = None
            
        print("[System] 런처 초기화 완료!")

    def _is_running(self) -> bool:
        return self.main_process is not None or self.camera_process is not None

    # ─────────────────────────────────────
    # 엔진 시작 (메인 엔진 -> 카메라 순서)
    # ─────────────────────────────────────
    def start_engine(self):
        with self._lock:
            if self._is_running():
                print("[System] 이미 엔진이 가동 중입니다.")
                return

            print("[System]주행 시작 신호 수신! 시스템 가동을 시작합니다...")

            # 1. 메인 엔진 실행 
            main_cmd = [
                MAIN_PYTHON, "main.py", 
                "--mode", "run",
                "--tflite", "./fer/models/efficientface_4cls_finetuned_fp16_float16.tflite",
                "--camera_backend", "tcp",
                "--tcp_host", CAMERA_HOST,
                "--tcp_port", str(CAMERA_PORT),
            ]
            self.main_process = subprocess.Popen(main_cmd, cwd=PROJECT_ROOT)
            print(f"[System] 메인 엔진 시작 (PID: {self.main_process.pid})")

            print("[System] 메인 엔진 초기화 및 서버 오픈 대기 중 (5초)...")
            time.sleep(5)

            # 2. 카메라 서비스 실행
            camera_cmd = [
                CAMERA_PYTHON, "fer/camera_service.py",
                "--host", CAMERA_HOST,
                "--port", str(CAMERA_PORT),
                "--width", "640",
                "--height", "480",
                "--fps", "20",
            ]
            self.camera_process = subprocess.Popen(camera_cmd, cwd=PROJECT_ROOT)
            print(f"[System] 카메라 서비스 시작 (PID: {self.camera_process.pid})")

            print("[System] 모든 시스템이 정상적으로 연결 및 가동되었습니다.")

    # ─────────────────────────────────────
    # 엔진 종료
    # ─────────────────────────────────────
    def stop_engine(self):
        with self._lock:
            if not self._is_running():
                print("[System]  현재 가동 중인 엔진이 없습니다.")
                return

            print("[System] 주행 종료 신호 수신! 시스템을 안전하게 종료합니다...")

            # 프로세스 종료 (자원 반환을 위해 메인 엔진 먼저 종료 권장)
            self._kill_process(self.main_process,   "메인 엔진")
            self._kill_process(self.camera_process, "카메라 서비스")

            self.main_process   = None
            self.camera_process = None

            print("[System] 종료 완료. 다음 주행 대기 중.")

           
    def _kill_process(self, proc: subprocess.Popen, name: str):
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=3)
            print(f"[System]  - {name} 종료 확인")
        except subprocess.TimeoutExpired:
            print(f"[System]  - {name} 응답 없음 → 강제 종료(kill)")
            proc.kill()
            proc.wait()

    # ─────────────────────────────────────
    # 실행 루프
    # ─────────────────────────────────────
    def run(self):
        print("[System] 앱 시그널 수신 대기 시작...")
        if self.ser:
            self._run_serial_loop()
        else:
            self._run_keyboard_loop()

    def _run_serial_loop(self):
        while True:
            try:
                line = self.ser.readline().decode("utf-8").strip()
                if not line: continue
                
                print(f"[System] 패킷 수신: '{line}'")
                if line == PACKET_START:
                    self.start_engine()
                elif line == PACKET_STOP:
                    self.stop_engine()
            except KeyboardInterrupt:
                self.stop_engine()
                break
            except Exception as e:
                print(f"[System] 통신 에러: {e}")
                break

    def _run_keyboard_loop(self):
        print("[System] 테스트 모드: 터미널에 start / stop / exit 입력")
        while True:
            try:
                cmd = input("Command >> ").strip().lower()
                if cmd == "start": self.start_engine()
                elif cmd == "stop": self.stop_engine()
                elif cmd == "exit":
                    if self._is_running():
                        self.stop_engine()
                    print("\n[System] 시스템을 완전히 종료합니다. 안녕히 가십시오!")
                    time.sleep(1) # 인사를 읽을 시간 1초 부여
                    break
            except KeyboardInterrupt:
                self.stop_engine()
                break

if __name__ == "__main__":
    launcher = SystemLauncher()
    launcher.run()