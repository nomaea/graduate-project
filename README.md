# FER Module

웹캠 기반 얼굴 감정 인식(FER) 모듈이다.
이 모듈은 얼굴 영상을 입력받아 감정을 추론하고, 결과를 **멀티모달 융합 엔진**으로 전달한다.

## Directory

```text
fer_module/
├─ fer_api.py
├─ fer_core.py
├─ fer_visualizer.py
└─ main.py
```

## Files

### `fer_api.py`

FER 결과를 외부 모듈과 연결하기 위한 **데이터 인터페이스 계층**이다.

주요 역할:

* FER 결과 packet 구조 정의
* queue publisher 제공
* 멀티모달 융합 엔진과의 전달 규약 정의

핵심 포인트:

* `FERPacket`: FER 결과 1건을 표현하는 데이터 구조
* `FERQueuePublisher`: 최신 FER 결과를 queue에 적재
* queue 정책: `Queue(maxsize=1)` 기반 최신값 우선

---

### `fer_core.py`

FER의 **핵심 추론 엔진**이다.

주요 역할:

* MediaPipe FaceMesh 기반 얼굴 landmark 검출
* 얼굴 정렬(alignment) 및 ROI crop
* TFLite 추론
* softmax 계산
* EMA smoothing
* FER packet 생성 및 queue publish

핵심 포인트:

* 클래스 순서 고정:

  ```python
  ["neutral", "happy", "sad", "angry"]
  ```
* 최종 출력값은 raw logits가 아니라 **softmax + EMA 적용 결과**
* `return_debug=True` 사용 시 얼굴 crop preview 반환 가능

---

### `fer_visualizer.py`

FER 결과를 확인하기 위한 **시각화/디버그 도구**이다.

주요 기능:

* FaceMesh landmark 표시
* landmark index 표시
* 영역별 landmark 색상 구분
* softmax panel 표시
* 현재 top1 감정 표시
* aligned face ROI 표시

용도:

* FER 동작 검증
* 발표/demo 시 시각적 확인
* landmark 기반 추론 설명

---

### `main.py`

시스템 실행을 위한 **entry point** 이다.

지원 모드:

* `run`: 운영 모드
* `visualize`: 시각화 모드

역할:

* 실행 모드 분기
* 운영 모드에서 FER/BIO/Fusion 제어
* 시각화 모드에서 `fer_visualizer.py` 실행

---

## Runtime Modes

### 1. Operation Mode

FER + BIO + Fusion 실행 모드

기능:

* 기본 웹캠 화면 출력
* 얼굴 crop preview 출력
* FER 결과 queue 전달
* BIO / Fusion 엔진 실행

실행:

```bash
python main.py --mode run --tflite efficientface_4cls_finetuned_fp16_float16.tflite
```

---

### 2. Visualization Mode

FER landmark UI 확인 모드

기능:

* landmark index 표시
* 영역별 landmark 색상 표시
* softmax panel 표시
* 현재 감정 표시
* aligned ROI 표시

실행:

```bash
python main.py --mode visualize --tflite efficientface_4cls_finetuned_fp16_float16.tflite
```

---

## Data Interface

### Queue Policy

FER는 producer, Fusion은 consumer 역할을 한다.

```python
Queue(maxsize=1)
```

정책:

* 오래된 packet 제거
* 최신 packet 유지

즉, 실시간성을 위해 **latest-value priority** 구조를 사용한다.

---

### Packet Format

```json
{
  "ts": 12345.678,
  "seq": 10,
  "class_order": ["neutral", "happy", "sad", "angry"],
  "softmax": [0.82, 0.10, 0.04, 0.04],
  "argmax_idx": 0,
  "argmax_label": "neutral",
  "source": "fer",
  "face_detected": true,
  "latency_ms": 38.4
}
```

---

### Class Order

FER 클래스 순서는 아래와 같이 고정되어 있다.

```python
["neutral", "happy", "sad", "angry"]
```

따라서:

* `softmax[0] = neutral`
* `softmax[1] = happy`
* `softmax[2] = sad`
* `softmax[3] = angry`

멀티모달 융합 엔진은 반드시 동일한 순서를 사용하거나, 명시적으로 remap해야 한다.

---

### Timestamp

timestamp는 `time.monotonic()` 기준이다.

이유:

* 시스템 시간 변경 영향 없음
* FER/BIO 간 시간 정렬에 유리
* 실시간 파이프라인에 적합

---

## Output Interpretation

FER 최종 출력은 다음 과정을 거친 값이다.

1. TFLite raw output 획득
2. `softmax(out)` 계산
3. EMA 적용
4. EMA 결과를 최종 softmax로 사용

즉, 화면/로그/queue에 전달되는 값은
**raw output이 아니라 softmax + EMA가 적용된 최종 확률값**이다.

---

## Visualization Keys

### Visualization Mode

* `q`: quit
* `i`: landmark index on/off
* `c`: landmark connection on/off
* `r`: aligned ROI on/off
* `l`: legend on/off

### Operation Mode

* `q`: quit

---

## Design Notes

이 구조는 다음 원칙으로 설계되었다.

* 추론 코어와 시각화 분리
* 운영 모드와 디버그 모드 분리
* 최신값 우선 queue 구조 유지
* 멀티모달 융합을 위한 packet 표준화
* timestamp / class order 명시적 고정

---

## Future Integration

현재 `main.py`의 `BIOEngineStub`, `FusionEngineStub`는 임시 구조이다.
실제 통합 시 아래 부분이 교체 대상이다.

* `BIOEngineStub` → 실제 BIO 엔진
* `FusionEngineStub` → 실제 멀티모달 융합 엔진

FER 측 핵심 연동 포인트:

* `fer_api.py`
* `fer_core.py`
* FER packet의 `ts`
* FER packet의 `softmax`
* 고정된 `class_order`

---

## Edge Device Python Environment

라즈베리파이 엣지 디바이스에서는 기본 시스템 Python 3.13 대신,  
`pyenv` 기반 **Python 3.11.11** 가상환경 `fer_facemesh311`을 별도로 구성하여 실행한다.

이유:

* `tflite-runtime`과 `mediapipe`의 호환성을 확보하기 위함
* FER/BIO/멀티모달 통합 실행 시 버전 충돌을 줄이기 위함
* Raspberry Pi 시스템 패키지와 추론용 Python 패키지를 분리 관리하기 위함

### Environment Summary

* Python: `3.11.11` (`pyenv` 기반)
* Virtual Environment: `fer_facemesh311`
* 실행 예시:

```bash
source ~/venvs/fer_facemesh311/bin/activate
python main.py --mode run --tflite ./fer/models/efficientface_4cls_finetuned_fp16_float16.tflite --camera_backend tcp --tcp_host 127.0.0.1 --tcp_port 9999
```

### Core Libraries and Versions

아래는 엣지 디바이스 통합 실행 기준 핵심 라이브러리이다.

| Category | Library | Version | Notes |
|---|---|---:|---|
| Runtime | Python | 3.11.11 | `pyenv` 기반 별도 구축 |
| FER Inference | `tflite-runtime` | 2.14.0 | TFLite 추론 엔진 |
| Landmark / FaceMesh | `mediapipe` | 0.10.18 | FaceMesh 기반 landmark 추출 |
| Numerical | `numpy` | 1.26.4 | 전처리 / softmax / 배열 연산 |
| MediaPipe Dependency | `jax` | 0.7.1 | MediaPipe 설치 시 함께 구성 |
| MediaPipe Dependency | `jaxlib` | 0.7.1 | MediaPipe 설치 시 함께 구성 |

### System Packages Used Together

카메라 입력은 별도 `camera_service.py`에서 처리하며, 이 부분은 Raspberry Pi OS의 시스템 패키지를 사용한다.
가상환경 내부 추론 프로세스와 분리된 이유는 `picamera2`/`libcamera` 계열 패키지가 시스템 Python과 더 강하게 결합되어 있기 때문이다.

주요 시스템 패키지:

* `python3-picamera2`
* `python3-libcamera`
* `python3-opencv`

즉, 현재 구조는 다음과 같이 분리된다.

* **camera_service.py** → 시스템 Python + Picamera2
* **main.py / FER / BIO / multimodal** → `pyenv` 기반 Python 3.11 가상환경

### Recommended Version Check Commands

실제 운영 직전에는 아래 명령으로 버전을 다시 확인하는 것을 권장한다.

```bash
source ~/venvs/fer_facemesh311/bin/activate
python -c "import sys; print(sys.version)"
python -c "import numpy; print('numpy', numpy.__version__)"
python -c "import mediapipe as mp; print('mediapipe', mp.__version__)"
python -c "import jax; print('jax', jax.__version__)"
python -c "import jaxlib; print('jaxlib', jaxlib.__version__)"
python -c "from tflite_runtime.interpreter import Interpreter; print('tflite-runtime OK')"
```

시스템 카메라 패키지는 별도로 확인한다.

```bash
python3 -c "from picamera2 import Picamera2; print('picamera2 OK')"
python3 -c "import cv2; print(cv2.__version__)"
```
