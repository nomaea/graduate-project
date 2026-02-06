# graduate-project
# Multimodal Stress Detection Prototype

표정(FER) + 생체신호(HRV/GSR)를 동시에 사용해서  
운전자의 **안전 상태 / 스트레스 상태**를 추정하고,  
그 결과를 **JSON + WebSocket**으로 모바일 앱에 보내는 프로토타입입니다.

---

## 1. 전체 구조

### 데이터 흐름

1. **카메라 / 센서 입력**

   - FER: 얼굴 표정 인식 모델 (TensorFlow Lite, OpenCV, MediaPipe 등)
   - Sensor: HRV, GSR 등 생체 신호

2. **멀티모달 엔진 (Python)**

   - FER 결과 + 센서 결과 → **Fusion Score** 계산
   - `safe`, `stressed` 확률 및 지배 상태(`dominant`) 산출
   - EMA(지수 이동 평균) 등으로 변동성 완화

3. **JSON 변환**

   - `fusion_result_to_json()` 을 통해 결과를 JSON 문자열로 변환

4. **전달 방식**

   - 콘솔 출력 (디버깅용) – `main.py`
   - WebSocket 서버 (모바일 연동용) – `main_ws.py`

5. **모바일 앱 (Android)**
   - WebSocket으로 JSON 수신
   - Gson으로 파싱 후 UI에 표시

---

## 2. 폴더 / 파일 구조

```text
Multimodal/
├── main.py                 # 콘솔 출력용 멀티모달 엔진 실행
├── main_ws.py              # WebSocket 서버 (JSON 실시간 전송)
│
├── multimodal_engine.py    # 멀티모달 Fusion 핵심 로직
├── multimodal_types.py     # FusionResult 데이터 구조 정의
├── json_builder.py         # FusionResult → JSON 문자열 변환
│
├── fer_interface.py        # FER 공용 인터페이스 (수정 X)
├── fer_types.py            # FER 결과 타입 정의
├── real_fer_source.py      # ★ 실제 FER 출력 연결 (update_frame 내부만 향후 수정)
│
├── sensor_interface.py     # Sensor 공용 인터페이스 (수정 X)
├── sensor_types.py         # Sensor 결과 타입 정의
├── real_sensor_source.py   # ★ 실제 BIO 데이터 연결 (update_window 내부만 향후 수정)
│
├── mock/                   # (초기 테스트용 - 나중에 제거 예정)
│   ├── fer_mock.py         # Mock FER Source
│   └── sensor_mock.py      # Mock Sensor Source
│
└── android_spec/           # Android JSON 파싱 스펙
    ├── FusionResponse.java
    ├── FusionParser.java
    ├── sample_json.txt
    └── example_output.txt



