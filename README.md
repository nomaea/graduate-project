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
multimodal/
│
├── main.py                        # 콘솔 실행 및 결과 확인용
├── main_ws.py                     # WebSocket 서버 (모바일 앱 송신용)
│
├── multimodal_engine.py           # 핵심 융합 엔진 (가중치 융합 + EMA + Alert)
├── multimodal_types.py            # FusionResult 데이터 타입
│
├── real_fer_source.py             # FER 큐에서 데이터 수신 및 변환
├── fer_interface.py               # FER 소스 추상 인터페이스
├── fer_types.py                   # FerResult 데이터 타입
│
├── real_sensor_source.py          # BIO 데이터 수신 및 변환
├── sensor_interface.py            # 센서 소스 추상 인터페이스
├── sensor_types.py                # SensorResult 데이터 타입
│
├── shared_queue.py                # FER/BIO 공유 큐
├── json_builder.py                # FusionResult → JSON 변환
├── test_edge_cases.py             # 단위 테스트



