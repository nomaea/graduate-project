# 코드 재설계 변경 내역

> 참고 문서: `ferProblem.md`, `mulProblem.md`
> 수정일: 2026-03-27

---

## 수정 파일 목록

| 파일 | 관련 이슈 |
|---|---|
| `fer/fer_core.py` | FER #1 #2 #3 #4 #5 #6 |
| `multimodal/fer_types.py` | MUL #5 |
| `multimodal/sensor_types.py` | MUL #5 |
| `multimodal/real_fer_source.py` | FER #7, MUL #1 |
| `multimodal/real_sensor_source.py` | MUL #4 #5 #7 |
| `multimodal/multimodal_engine.py` | MUL #1 #2 #3 #5 #6 #7 #9 #10 |
| `bio/bio_ppg_processor.py` | FER #8 |
| `main.py` | FER #9 #10, MUL #8 |

---

## FER 모듈 수정 내역

### 🔴 FER#1 — TFLite 추론 예외 처리 추가
**파일:** `fer/fer_core.py`

```python
# 수정 전
self.interpreter.set_tensor(...)
self.interpreter.invoke()
out = self.interpreter.get_tensor(...)[0]

# 수정 후
try:
    self.interpreter.set_tensor(...)
    self.interpreter.invoke()
    out = self.interpreter.get_tensor(...)[0]
    probs = softmax(out)
except Exception as exc:
    _logger.warning("[FER] TFLite inference failed, skipping frame: %s", exc)
    probs = None
```

`interpreter.invoke()` 실패 시 해당 프레임을 skip하고 로그만 남김.
프로세스 크래시 방지.

---

### 🔴 FER#2 — 얼굴 미감지 시 `prob_ema` 리셋
**파일:** `fer/fer_core.py`

```python
# 추가된 상수 및 필드
NO_FACE_RESET_FRAMES = 5
self._no_face_count = 0  # __init__에 추가

# 로직 (process_frame)
if probs_out is None:
    self._no_face_count += 1
    if self._no_face_count >= NO_FACE_RESET_FRAMES and self.prob_ema is not None:
        self.prob_ema = None  # 이전 사람의 감정 잔류 방지
```

5프레임 이상 얼굴이 없으면 EMA를 초기화하여 새로운 사람이 등장했을 때 이전 사람의 감정 확률로 시작하는 문제 해결.

---

### 🔴 FER#3 — 비정방형 ROI 왜곡 방지
**파일:** `fer/fer_core.py`

```python
# 추가된 함수
def _pad_to_square(roi: np.ndarray) -> np.ndarray:
    h, w = roi.shape[:2]
    if h == w:
        return roi
    side = max(h, w)
    padded = np.zeros((side, side, 3), dtype=roi.dtype)
    pad_y = (side - h) // 2
    pad_x = (side - w) // 2
    padded[pad_y:pad_y + h, pad_x:pad_x + w] = roi
    return padded

# get_mesh_aligned_roi 내부 마지막에 적용
roi_bgr = rotated[y1:y2, x1:x2]
roi_bgr = _pad_to_square(roi_bgr)  # 추가
```

프레임 경계 클램핑으로 ROI가 직사각형이 되더라도 제로 패딩으로 정방형을 유지.
`cv2.resize(→ 224×224)` 시 얼굴 왜곡 방지.

---

### 🟠 FER#4 — EMA 계수 조정
**파일:** `fer/fer_core.py`

```python
# 수정 전
PROB_EMA = 0.65  # 새 값 반영 비율 35%

# 수정 후
PROB_EMA = 0.4   # 새 값 반영 비율 60%
```

`0.65^n ≈ 0.05` 기준으로 이전 값이 5% 미만으로 감쇠되는 데 약 8프레임이 필요했던 것을 약 4프레임으로 단축.

---

### 🟠 FER#5 — `process_frame` 반환 타입 통일
**파일:** `fer/fer_core.py`

```python
# 수정 전: 얼굴 없을 때 return None (return_debug=False)
# 수정 후: 항상 dict 반환
return {"packet": None, "roi_preview": None, "face_detected": False}
```

두 경로 모두 항상 `dict`를 반환하도록 통일.
`return_debug` 모드 전환 시 `TypeError` 발생 가능성 제거.

---

### 🟠 FER#6 — 타임스탬프 기준 통일
**파일:** `fer/fer_core.py`

```python
# 수정 전
frame_ts = time.monotonic()  # 부팅 기준 단조시계

# 수정 후
frame_ts = time.time()       # Unix epoch 기준
```

BIO 모듈의 `time.time()` 기준과 일치시켜 두 모달리티의 시간 정렬 가능.

---

### 🟠 FER#7 — `face_detected` 기본값 수정
**파일:** `multimodal/real_fer_source.py`

```python
# 수정 전
face_detected = packet.get("face_detected", True)

# 수정 후 (로직 자체를 MUL#1과 함께 제거 — 아래 참고)
```

기본값 `True`가 유발하는 얼굴 없는 상황을 "있는 것"으로 오판하는 문제 해결.

---

### 🟡 FER#8 — `_detect_peak` 최소 길이 강화
**파일:** `bio/bio_ppg_processor.py`

```python
# 수정 전
if len(self.ir_filt) < 5:
    return

# 수정 후
if len(self.ir_filt) < 10:
    return
```

배치 입력 도중 `_reset_keep_raw()` 호출 후 `ir_filt` 재충전 중에 이전·이후 데이터가 혼합되어 피크를 오감지하는 경우 방지.

---

### 🟡 FER#9 — 창 생성 상태 플래그 추가
**파일:** `main.py`

```python
# 수정 전: getWindowProperty로 창 존재 여부를 매 프레임 확인
try:
    if cv2.getWindowProperty("Face Crop Preview", cv2.WND_PROP_VISIBLE) >= 0:
        cv2.destroyWindow("Face Crop Preview")
except cv2.error:
    pass

# 수정 후: 플래그로 상태 추적
if self._crop_window_open:
    cv2.destroyWindow("Face Crop Preview")
    self._crop_window_open = False
```

`self._crop_window_open = False`를 `__init__`에 추가하고 창 생성·소멸 시 갱신.
매 프레임 불필요한 예외 발생 제거.

---

### 🟡 FER#10 — DEBUG print → logging
**파일:** `main.py`

```python
# 수정 전
print(f"[DEBUG][BIO_QUEUE] drained={drained} bpm=...")

# 수정 후
logging.debug("[BIO_QUEUE] drained=%d bpm=%s ...", drained, ...)
```

30fps 기준 초당 120줄 이상의 콘솔 출력을 `logging.debug()`로 교체.
운영 환경에서 `logging.INFO` 수준으로 실행하면 자동 비활성화.

---

## Multimodal 모듈 수정 내역

### 🔴 MUL#1 — 졸음 감지 로직 교정
**파일:** `multimodal/real_fer_source.py`, `multimodal/multimodal_engine.py`

```python
# 수정 전 (real_fer_source.py)
face_detected = packet.get("face_detected", True)
drowsy_scores = {
    "alert": 1.0 if face_detected else 0.0,
    "drowsy": 0.0 if face_detected else 1.0,
}

# 수정 후
# 얼굴 유무로 drowsy를 결정하는 로직 완전 제거
# 실제 EAR(Eye Aspect Ratio) 구현 전까지 중립값 고정
drowsy_scores = {"alert": 0.5, "drowsy": 0.5}
```

```python
# 수정 전 (multimodal_engine.py _decide_alert)
if ema_stress >= 0.75 or drowsy_val >= 0.60:  # 고개 돌림만으로도 DANGER
    level = "DANGER"

# 수정 후
if ema_stress >= 0.75:  # drowsy_val 조건 제거
    level = "DANGER"
```

| 기존 오동작 사례 | 수정 후 |
|---|---|
| 눈 반쯤 감긴 졸음 → alert=1.0 | 중립 처리 |
| 고개를 잠깐 돌림 → DANGER | 스트레스 EMA 기반으로만 판단 |
| 조명 부족 얼굴 미감지 → DANGER | 중립 처리 |

---

### 🔴 MUL#2 — Degraded Mode 추가
**파일:** `multimodal/multimodal_engine.py`

```python
# 수정 전
if fer_res is None or sensor_res is None:
    return None  # 둘 다 있어야만 동작

# 수정 후
if fer_res is None and sensor_res is None:
    return None

degraded = False
if sensor_res is None:
    sensor_res = self._make_neutral_sensor()  # BIO 중립 더미
    degraded = True
elif fer_res is None:
    fer_res = self._make_neutral_fer()         # FER 중립 더미
    degraded = True
```

- **BIO-only**: 시스템 시작 후 BIO 준비 전(30~40초) 또는 센서 연결 실패 시에도 FER 결과 출력
- **FER-only**: FER 모델 로드 실패 또는 카메라 없을 때도 BIO 결과 출력
- Degraded 모드에서 `confidence × 0.5` 적용하여 신뢰도 표현

---

### 🔴 MUL#3 — `_decide_alert` side effect 제거
**파일:** `multimodal/multimodal_engine.py`

```python
# 수정 전: _decide_alert 내부에서 self.smoothed_stress 변경
def _decide_alert(self, fused_scores, fer):
    ema_stress = self._smooth_stress(raw_stressed)  # ← state 변경!
    ...

# 수정 후: step()에서 명시적으로 호출, 결과를 인자로 전달
def step(self):
    ...
    ema_stress = self._smooth_stress(fused_scores["stressed"])  # step()에서 호출
    alert_level, alert_reason = self._decide_alert(ema_stress)  # 값만 전달

def _decide_alert(self, ema_stress: float) -> Tuple[str, str]:
    ...
```

동일 입력으로 `step()`을 두 번 호출해도 EMA가 두 번 적용되지 않음. 재현성 보장.

---

### 🟠 MUL#4 — 낮은 BIO 신호 품질 시 중립 처리
**파일:** `multimodal/real_sensor_source.py`

```python
# 수정 전
stressed = bio_arousal * bio_conf   # conf=0.1이면 stressed≈0, safe≈1.0 (오류)
safe = 1.0 - stressed

# 수정 후
_LOW_CONF_THRESHOLD = 0.3
if bio_conf < _LOW_CONF_THRESHOLD:
    stressed = 0.5  # 신뢰 불가 → 중립
    safe = 0.5
else:
    stressed = bio_arousal * bio_conf
    safe = 1.0 - stressed
```

손가락 미접촉 등으로 신호 품질이 낮을 때 Fusion에서 60% 가중치로 "매우 안전" 신호를 보내던 문제 해결.

---

### 🟠 MUL#5 — Fusion 타임스탬프 관측 시점 기준으로 수정
**파일:** `multimodal/fer_types.py`, `multimodal/sensor_types.py`, `multimodal/multimodal_engine.py`

```python
# fer_types.py / sensor_types.py에 ts 필드 추가
@dataclass
class FerResult:
    ...
    ts: float = 0.0  # 관측 시점 Unix 타임스탬프

# multimodal_engine.py
# 수정 전
return FusionResult(timestamp=time.time(), ...)  # step() 완료 시점

# 수정 후
fer_ts = fer_res.ts if fer_res.ts > 0 else time.time()
sensor_ts = sensor_res.ts if sensor_res.ts > 0 else time.time()
obs_timestamp = min(fer_ts, sensor_ts)  # 더 오래된 소스 시점 사용
return FusionResult(timestamp=obs_timestamp, ...)
```

BIO는 최대 1초 전 데이터임에도 FusionResult 타임스탬프가 현재 시각으로 기록되던 문제 해결.

---

### 🟠 MUL#6 — 가중치 합산 검증
**파일:** `multimodal/multimodal_engine.py`

```python
# __init__ 내부
total = w_fer + w_sensor
if abs(total - 1.0) > 1e-6:
    _logger.warning(
        "[MultiModal] w_fer(%.3f) + w_sensor(%.3f) = %.4f ≠ 1.0, normalizing.",
        w_fer, w_sensor, total,
    )
    w_fer    /= total
    w_sensor /= total
self.w_fer    = w_fer
self.w_sensor = w_sensor
```

`w_fer=0.3, w_sensor=0.3`처럼 합이 1이 아닌 값이 들어와도 조용히 잘못된 가중치로 동작하던 문제 해결.

---

### 🟠 MUL#7 — Confidence에 신호 품질 반영
**파일:** `multimodal/multimodal_engine.py`

```python
# 수정 전
confidence = fused_scores[dominant]  # 0.5~1.0 범위, 신호 품질 미반영

# 수정 후
bio_conf    = sensor_res.raw_metrics.get("bio_conf", 1.0)
fer_quality = max(fer_res.emotion_scores.values())  # 최고 softmax 확률
signal_quality = min(bio_conf, fer_quality) if not degraded else 0.5
confidence = fused_scores[dominant] * signal_quality
```

GSR 미연결·손가락 미접촉 상황에서도 confidence가 높게 표현되던 문제 해결.

---

### 🟠 MUL#8 — `[FUSION_JSON]` print → logging
**파일:** `main.py`

```python
# 수정 전
print("[FUSION_JSON]", fusion_json_str)  # 매 프레임 전체 JSON 출력

# 수정 후
logging.debug("[FUSION_JSON] %s", fusion_json_str)
```

Fusion 결과가 카메라 프레임마다 발생할 경우 초당 30회 JSON print가 발생하던 I/O 부하 제거.

---

### 🟡 MUL#9 — FER→stress 변환 수식 균형 수정
**파일:** `multimodal/multimodal_engine.py`

```python
# 수정 전: drowsy 이진값(0 또는 1)이 수식 불균형 유발
stressed = 0.6 * (sad + angry) + 0.4 * drowsy_val
safe     = 0.6 * (happy + neutral) + 0.4 * alert
# → 얼굴 있을 때 safe 최솟값이 0.4에서 시작

# 수정 후: 순수 감정 점수만 사용
stressed = sad + angry
safe     = happy + neutral
```

실제 EAR 기반 졸음 감지 구현 시 drowsy 항 재추가 가능하도록 주석으로 명시.

---

### 🟡 MUL#10 — Python 3.8 호환 타입 힌트
**파일:** `multimodal/multimodal_engine.py`

```python
# 수정 전 (Python 3.9+ 전용)
from typing import Dict, Optional
def _decide_alert(...) -> tuple[str, str]:

# 수정 후 (Python 3.8 호환)
from typing import Dict, Optional, Tuple
def _decide_alert(...) -> Tuple[str, str]:
```

Raspberry Pi OS 기본 Python 3.8 환경에서 임포트 시 크래시 방지.

---

## 향후 과제

- **실제 졸음 감지 구현**: EAR(Eye Aspect Ratio) 기반 눈 감김 판별 → `drowsy_scores`에 연결
- **FER 신호 품질 지표 개선**: 단순 최고 softmax 확률 대신 entropy 기반 불확실성 측정 고려
- **BIO 초기화 대기 시간 단축**: `RMSSD_MIN_RR=12` 파라미터 튜닝 또는 단계적 신뢰도 상승 방식 검토
