# EdgeDevice — Real-Time Multimodal Emotion Recognition Engine

엣지 디바이스(Raspberry Pi / Jetson 계열)에서 구동되는 실시간 멀티모달 감정·스트레스 인식 시스템입니다.  
카메라(FER)와 생체 센서(BIO)를 융합하여 운전자 또는 사용자의 스트레스·졸음 상태를 1초 단위로 분석하고, WebSocket으로 결과를 전송합니다.

---

## 목차

- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [모듈 설명](#모듈-설명)
- [하드웨어 요구사항](#하드웨어-요구사항)
- [소프트웨어 의존성](#소프트웨어-의존성)
- [실행 방법](#실행-방법)
- [출력 형식](#출력-형식)
- [알림 로직](#알림-로직)
- [융합 공식](#융합-공식)
- [로그 및 메트릭](#로그-및-메트릭)
- [디렉토리 구조](#디렉토리-구조)

---

## 주요 기능

| 기능 | 세부 내용 |
|---|---|
| **표정 인식 (FER)** | MediaPipe FaceMesh + EfficientFace TFLite → neutral / happy / sad / angry |
| **졸음 감지** | EAR(Eye Aspect Ratio) 기반 — 눈 감김 2초 이상 지속 시 졸음 판정 |
| **생체 신호 (BIO)** | MAX30102 PPG (BPM, HRV/RMSSD) + 시리얼 GSR → 생리적 각성도 산출 |
| **멀티모달 융합** | FER 40% + BIO 60% 가중 결합 + EMA 스무딩 |
| **3단계 경보** | NORMAL / WARNING / DANGER 실시간 판별 |
| **WebSocket 출력** | `ws://0.0.0.0:8765` — JSON 페이로드 1초 주기 브로드캐스트 |
| **CSV 로깅** | `logs/bio/YYYYMMDD/` 경로에 생체 신호 전체 기록 |
| **Degraded Mode** | FER 또는 BIO 중 하나만 살아 있어도 부분 결과 출력 (신뢰도 70%) |
| **HTTP 제어 서버** | `pi_server.py` — `POST /start`, `POST /stop` + 세션 메트릭 반환 |

---

## 시스템 아키텍처

```
┌────────────────────────────────────────────────────────┐
│                       main.py                          │
│                   (RuntimeController)                  │
│                                                        │
│  ┌──────────────────┐    ┌────────────────────────┐   │
│  │   FER Pipeline   │    │     BIO Pipeline        │   │
│  │  (camera thread) │    │   (background thread)   │   │
│  │                  │    │                          │   │
│  │  TCP stream       │    │  MAX30102 (I2C, 100 Hz)  │   │
│  │      ↓            │    │  GSR Serial (auto-detect)│   │
│  │  FERCore          │    │  PPGProcessor            │   │
│  │  MediaPipe +      │    │  (BPM, RMSSD, GSR)       │   │
│  │  TFLite Inference │    │      ↓ 1 Hz              │   │
│  │      ↓            │    │  payload_queue           │   │
│  │  FERQueuePublisher│    └────────────────────────┘   │
│  │      ↓            │              ↓                  │
│  │   fer_queue       │       RealSensorSource          │
│  └──────────────────┘              ↓                  │
│          ↓                                             │
│     RealFerSource                                      │
│          ↓                                             │
│   ┌──────────────────────────────────────────┐        │
│   │          MultiModalEngine.step()          │        │
│   │   FER 40%  +  BIO 60%  →  EMA Fusion     │        │
│   │   NORMAL / WARNING / DANGER 판별           │        │
│   └──────────────────────────────────────────┘        │
│          ↓                    ↓                        │
│   WebSocket Broadcast    CSV Logging                   │
│   ws://0.0.0.0:8765                                   │
└────────────────────────────────────────────────────────┘
```

### 프로세스 구성 (pi_server 사용 시)

```
pi_server.py (HTTP :5000)
├── main.py          — 멀티모달 엔진 + WebSocket
└── camera_service.py — 카메라 캡처 → TCP :9999 전송
```

---

## 모듈 설명

### `fer/` — 표정 인식

| 파일 | 역할 |
|---|---|
| `fer_core.py` | MediaPipe FaceMesh → 눈 정렬 ROI → EfficientFace TFLite 추론 → EMA 스무딩 (α=0.4) |
| `fer_api.py` | `FERPacket` 데이터 클래스, `FERQueuePublisher` (Queue 발행) |
| `camera_service.py` | v4l2 / OpenCV 카메라 캡처 → JPEG 압축 → TCP 전송 |
| `fer_visualizer.py` | 시각화 유틸리티 (디버그용) |

**FER 처리 과정:**
1. `camera_service.py` → JPEG 프레임을 TCP로 전송 (기본 포트 9999)
2. `main.py`의 TCP 서버 스레드가 수신 → `FERCore.process_frame()` 호출
3. MediaPipe FaceMesh로 468개 랜드마크 검출
4. 양쪽 눈 외안각 기준으로 얼굴 정렬 후 224×224 ROI 추출
5. TFLite 추론 → Softmax → EMA 스무딩
6. EAR 계산 → 2초 이상 눈 감김 시 `is_drowsy=True`
7. `FERPacket`을 `fer_queue`에 발행

### `bio/` — 생체 신호 처리

| 파일 | 역할 |
|---|---|
| `bio_engine_core.py` | 메인 루프 (`main(payload_queue)`) — 100 Hz 샘플링, 1 Hz 출력 |
| `bio_ppg_processor.py` | PPG 피크 감지, BPM EMA, 30초 RMSSD 계산, RR 검증 |
| `bio_gsr_reader.py` | 시리얼 GSR 비동기 판독 (`/dev/ttyUSB*`, `/dev/ttyACM*` 자동 탐색) |
| `max30102_better.py` | MAX30102 I2C 드라이버 |
| `bio_utils.py` | `clamp01`, `norm_range`, `mean` 등 유틸리티 |

**BIO 각성도(Arousal) 산출 가중치:**

| 신호 | 가중치 | 방향 |
|---|---|---|
| BPM | 35% | 높을수록 각성 ↑ (범위: 55~120) |
| RMSSD (HRV) | 30% | 낮을수록 각성 ↑ (범위: 0.01~0.12 s) |
| GSR 기울기 (1s) | 20% | 양수 기울기 = 각성 ↑ |
| GSR 피크 (10s) | 15% | 빈도 높을수록 각성 ↑ |

### `multimodal/` — 융합 엔진

| 파일 | 역할 |
|---|---|
| `multimodal_engine.py` | FER + BIO 가중 융합, EMA 스트레스, 알림 판별 |
| `real_fer_source.py` | `fer_queue`에서 FER 결과 드레인 → `FerResult` |
| `real_sensor_source.py` | `payload_queue`에서 BIO 결과 업데이트 → `SensorResult` |
| `ws_server.py` | AsyncIO WebSocket 서버 (백그라운드 스레드) |
| `json_builder.py` | `FusionResult` → JSON 직렬화 |
| `multimodal_types.py` | `FusionResult` 데이터 클래스 |

### `pi_server.py` — HTTP 제어 서버

Raspberry Pi에서 엔진 프로세스를 원격으로 시작/종료하는 HTTP 서버 (포트 5000).

| 엔드포인트 | 메서드 | 동작 |
|---|---|---|
| `/start` | POST | `main.py` + `camera_service.py` 프로세스 기동, 모니터링 시작 |
| `/stop` | POST | 프로세스 종료 후 세션 메트릭 JSON 반환 |

---

## 하드웨어 요구사항

- **보드**: Raspberry Pi 4 / 5 또는 Jetson Nano 계열 (Linux ARM/x86)
- **카메라**: V4L2 호환 USB 카메라 또는 Pi Camera
- **PPG 센서**: MAX30102 (I2C 버스)
- **GSR 센서**: 아날로그 → USB 시리얼 변환기 (`/dev/ttyUSB0` 또는 `/dev/ttyACM0` 자동 탐색)

---

## 소프트웨어 의존성

```
opencv-python
numpy
mediapipe
tflite-runtime        # 또는 tensorflow (x86 개발 환경)
websockets
pyserial
smbus2                # MAX30102 I2C
scipy
psutil                # pi_server.py CPU/메모리 모니터링 (선택)
```

설치 예시:

```bash
pip install opencv-python numpy mediapipe tflite-runtime websockets pyserial smbus2 scipy psutil
```

> **Raspberry Pi**: `tflite-runtime`은 [공식 휠](https://github.com/google-coral/pycoral/releases)에서 아키텍처에 맞는 버전을 설치하세요.

---

## 실행 방법

### 1. 직접 실행 (TCP 카메라 모드)

```bash
# 터미널 1: 메인 엔진 시작
python main.py \
  --camera_backend tcp \
  --tcp_host 127.0.0.1 \
  --tcp_port 9999 \
  --ws_port 8765

# 터미널 2: 카메라 서비스 시작
python fer/camera_service.py \
  --host 127.0.0.1 \
  --port 9999 \
  --width 640 \
  --height 480 \
  --fps 20
```

### 2. HTTP 서버를 통한 원격 제어

```bash
# Pi 서버 시작
python pi_server.py

# 엔진 시작
curl -X POST http://<pi-ip>:5000/start

# 엔진 종료 및 세션 메트릭 수신
curl -X POST http://<pi-ip>:5000/stop
```

### 3. 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--camera_backend` | `tcp` | 카메라 입력 방식 (`tcp` 고정) |
| `--tcp_host` | `127.0.0.1` | TCP 카메라 서버 호스트 |
| `--tcp_port` | `9999` | TCP 카메라 서버 포트 |
| `--ws_port` | `8765` | WebSocket 출력 포트 |
| `--tflite` | `fer/models/efficientface_4cls_finetuned_fp16_float16.tflite` | TFLite 모델 경로 |

---

## 출력 형식

### WebSocket JSON 페이로드 (1초 주기)

```json
{
  "timestamp": 1746500000.123,
  "fusion": {
    "dominant": "stressed",
    "confidence": 0.72,
    "scores": {
      "safe": 0.38,
      "stressed": 0.62
    }
  },
  "alert": {
    "level": "WARNING",
    "reason": "Stress level is elevated (EMA=0.53)"
  },
  "fer": {
    "emotion_scores": {
      "neutral": 0.10,
      "happy": 0.05,
      "sad": 0.45,
      "angry": 0.40
    },
    "is_drowsy": false,
    "ear": 0.28,
    "face_detected": true
  },
  "sensor": {
    "bpm": 88.3,
    "rmssd_use": 0.042,
    "bio_arousal": 0.71,
    "bio_conf": 0.85,
    "gsr_cur": 142.5,
    "finger_on": true,
    "gsr_fresh": true
  }
}
```

---

## 알림 로직

```
ema_stress >= 0.75  또는  drowsy_val >= 0.60  →  DANGER
ema_stress >= 0.50                            →  WARNING
그 외                                         →  NORMAL

※ face_missing == 1.0  →  WARNING (카메라 미감지 상태)
```

**EMA 스트레스** (`ema_alpha=0.3`):
```
ema_stress(t) = 0.3 × raw_stress(t) + 0.7 × ema_stress(t-1)
```

**Degraded Mode**: FER 또는 BIO 중 하나가 없을 경우, 나머지 모달리티만으로 결과를 산출하되 `confidence × 0.7` 패널티를 적용합니다.

---

## 융합 공식

```python
# FER → stressed / safe
stressed_fer = 0.6 * (sad + angry) + 0.4 * drowsy_val
safe_fer     = 0.6 * (happy + neutral) + 0.4 * alert_val

# BIO → stressed / safe  (신뢰도 < 0.3 시 중립 0.5 고정)
stressed_bio = bio_arousal * bio_conf
safe_bio     = 1.0 - stressed_bio

# 가중 융합 (FER 40% + BIO 60%)
safe_fused     = 0.4 * safe_fer     + 0.6 * safe_bio
stressed_fused = 0.4 * stressed_fer + 0.6 * stressed_bio
```

---

## 로그 및 메트릭

### BIO CSV 로그

경로: `logs/bio/YYYYMMDD/bio_v2p5_HHMMSS.csv`

| 컬럼 | 설명 |
|---|---|
| `ts_iso` | ISO 8601 타임스탬프 |
| `bpm` | 심박수 (EMA 스무딩) |
| `rmssd_ema_s` | RMSSD EMA (초 단위) |
| `gsr` | 현재 GSR 값 |
| `gsr_slope_1s` | GSR 1초 기울기 |
| `gsr_peak_10s` | GSR 10초 내 피크 횟수 |
| `bio_arousal` | 생리적 각성도 (0~1) |
| `bio_conf` | 신호 품질 신뢰도 (0~1) |
| `finger_on` | 손가락 접촉 여부 |
| `note` | 이상 상태 메모 (`NO_FINGER`, `LOW_PPG_Q`, `NO_GSR`) |

### FER 세션 메트릭

종료 시 `/tmp/fer_main_metrics.json`에 저장:

```json
{
  "session_duration_s": 120.5,
  "fer_frame_count": 2410,
  "avg_fps": 20.0,
  "avg_latency_ms": 18.3,
  "min_latency_ms": 12.1,
  "max_latency_ms": 45.7
}
```

---

## 디렉토리 구조

```
EdgeDevice/
├── main.py                  # 메인 실행 진입점
├── pi_server.py             # HTTP 원격 제어 서버
├── fer/
│   ├── fer_core.py          # FER 추론 코어 (MediaPipe + TFLite + EAR)
│   ├── fer_api.py           # FERPacket, FERQueuePublisher
│   ├── camera_service.py    # 카메라 캡처 → TCP 전송
│   ├── fer_visualizer.py    # 디버그 시각화
│   └── models/
│       └── efficientface_4cls_finetuned_fp16_float16.tflite
├── bio/
│   ├── bio_engine_core.py   # BIO 메인 루프
│   ├── bio_ppg_processor.py # PPG 신호 처리 (BPM, RMSSD)
│   ├── bio_gsr_reader.py    # GSR 시리얼 판독
│   ├── max30102_better.py   # MAX30102 I2C 드라이버
│   └── bio_utils.py         # 유틸리티
├── multimodal/
│   ├── multimodal_engine.py # 융합 엔진
│   ├── real_fer_source.py   # FER 큐 소비자
│   ├── real_sensor_source.py# BIO 큐 소비자
│   ├── ws_server.py         # WebSocket 서버
│   ├── json_builder.py      # JSON 직렬화
│   └── multimodal_types.py  # FusionResult 타입
└── logs/
    └── bio/
        └── YYYYMMDD/        # BIO CSV 로그
```
