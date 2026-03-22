# graduate-project
캡5 팀 졸업합시다

BIO Engine - Multimodal Sensor Input README

1. 개요

이 문서는 현재 바이오 엔진 구조와, 멀티모달 쪽에서 수정한 부분을 정리한 문서다.

현재 목표는 라즈베리파이 내부에서 바이오 엔진이 계산한 결과를 멀티모달 입력부로 전달하는 것이다.
이 단계에서는 FER 융합이나 최종 추론 결과 출력까지 포함하지 않고, BIO -> 멀티모달 입력 전달까지만 다룬다.

2. 현재 바이오 엔진 구조

현재 바이오 엔진은 다음과 같이 분리되어 있다.

- `bio_engine_v2p5_realtime.py`
  - 실행 진입 파일
  - 실제 연산은 직접 하지 않고 `bio_engine_core.py`를 호출한다.

- `bio_engine_core.py`
  - 전체 실행 루프를 담당한다.
  - MAX30102 샘플 읽기
  - GSR 시리얼 값 읽기
  - BPM, RMSSD, GSR slope, GSR peak 계산
  - bio_arousal, bio_conf 계산
  - CSV 로그 저장
  - payload 생성
  - queue를 통해 payload 전달

- `bio_gsr_reader.py`
  - GSR 시리얼 포트 탐색 및 읽기
  - 최신 GSR 값과 timestamp 보관

- `bio_ppg_processor.py`
  - PPG 샘플 처리
  - peak 검출
  - RR interval 계산
  - BPM 계산
  - RMSSD 계산
  - signal quality 계산

- `bio_utils.py`
  - 바이오 엔진 전반에서 공통으로 사용하는 보조 함수들을 모아둔 파일
  - 값 정규화, 시간 문자열 생성, 평균·중앙값 같은 기본 계산을 여기서 처리
  - clamp01
  - norm_range
  - now_iso
  - mean
  - stdev
  - median
  - mad

3. 바이오 엔진에서 계산하는 값

바이오 엔진은 1초마다 아래 값들을 계산한다.

- `bio_arousal`
- `bio_conf`
- `bpm`
- `rmssd_use`
- `gsr_cur`
- `gsr_slope_1s`
- `gsr_peak_10s`
- `ppgQ`
- `finger_on`
- `gsr_fresh`
- `note`
- `timestamp`

의미는 다음과 같다.

- `bio_arousal`
  - 현재 바이오 엔진의 최종 각성도 값
  - BPM, RMSSD 역방향 값, GSR slope, GSR peak를 이용해 계산한다.

- `bio_conf`
  - 현재 결과를 얼마나 신뢰할 수 있는지 나타내는 값
  - 손가락 접촉 여부, GSR 최신성, PPG quality가 반영된다.

- `bpm`
  - 현재 심박수

- `rmssd_use`
  - HRV 대표값
  - RMSSD raw와 EMA를 반영해 실제 사용값으로 쓴다.

- `gsr_cur`
  - 현재 GSR 대표값

- `gsr_slope_1s`
  - 최근 1초 기준 GSR 변화량

- `gsr_peak_10s`
  - 최근 10초 내 GSR 반응 피크 개수

- `ppgQ`
  - PPG 신호 품질 점수

- `finger_on`
  - 손가락 접촉 여부

- `gsr_fresh`
  - GSR 값이 최근에 정상적으로 갱신되었는지 여부

- `note`
  - 상태 메모
  - 예: `finger_off`, `hrv_warming_up(need~30s)`

4. 바이오 엔진이 멀티모달에 넘겨주는 값

현재 바이오 엔진이 멀티모달 쪽으로 넘기는 payload는 아래 형식이다.

python
payload = {
    "timestamp": now_iso(),
    "bio_arousal": float(bio_arousal),
    "bio_conf": float(bio_conf),
    "bpm": float(bpm) if bpm is not None else None,
    "rmssd_use": float(rmssd_use) if rmssd_use is not None else None,
    "gsr_cur": float(gsr_cur) if gsr_cur is not None else None,
    "gsr_slope_1s": float(gsr_slope_1s) if gsr_slope_1s is not None else None,
    "gsr_peak_10s": int(gsr_peak_10s),
    "ppgQ": float(ppg_q),
    "finger_on": bool(finger_on),
    "gsr_fresh": bool(gsr_fresh),
    "note": note,
}


이 payload는 `bio_engine_core.py`에서 생성된다.

5. 전달 방식

현재 전달 방식은 queue 기반이다.

흐름은 아래와 같다.

1. `bio_engine_core.py`가 1초마다 payload 생성
2. payload를 `Queue`에 넣음
3. `run_bio_to_multimodal_queue.py`가 queue에서 최신 payload를 꺼냄
4. `multimodal/real_sensor_source.py`의 `update_window(payload)`에 전달
5. `RealSensorSource`가 `SensorResult` 형태로 변환하여 내부에 저장
6. 멀티모달 엔진은 이후 `get_latest_result()`를 통해 센서 입력을 읽을 수 있음

현재는 3번까지가 실행되고 있고, `get_latest_result()`로 실제 결과가 생성되는 것까지 확인했다.

6. 멀티모달 쪽에서 수정한 부분

원래 `real_sensor_source.py`는 예전 바이오 입력 구조를 기준으로 만들어져 있었고, 내부에서 `BioService`를 다시 호출하는 구조였다.

즉 원래는 다음 흐름이었다.

- payload 입력
- `BioService.process_window(payload)` 실행
- 예전 입력 구조 기준 특징 추출 및 점수 계산
- `SensorResult` 생성

현재는 바이오 엔진이 이미 최종 특징값과 추론값을 계산해서 전달하므로, 이 구조는 중복이다.

그래서 현재 `real_sensor_source.py`는 다음 방식으로 바꿨다.

- `BioService` 의존 제거
- 바이오 엔진 payload 키를 직접 읽음
- `SensorResult(raw_metrics, emotion_scores)` 구조는 유지
- `update_window(payload)` 함수 이름 유지
- `get_latest_result()` 함수 이름 유지

즉 멀티모달 쪽 구조는 최대한 유지하고, 입력만 현재 바이오 엔진 기준으로 맞췄다.

7. real_sensor_source.py에서 사용하는 매핑

현재 `real_sensor_source.py`는 바이오 payload를 받아 아래처럼 저장한다.

raw_metrics

- `bpm` <- `payload["bpm"]`
- `hrv` <- `payload["rmssd_use"]`
- `gsr` <- `payload["gsr_cur"]`
- `bio_arousal` <- `payload["bio_arousal"]`
- `bio_conf` <- `payload["bio_conf"]`
- `gsr_slope_1s` <- `payload["gsr_slope_1s"]`
- `gsr_peak_10s` <- `payload["gsr_peak_10s"]`
- `ppgQ` <- `payload["ppgQ"]`
- `finger_on` <- `payload["finger_on"]`
- `gsr_fresh` <- `payload["gsr_fresh"]`

emotion_scores

현재는 아래처럼 사용한다.

- `stressed = bio_arousal`
- `safe = 1.0 - stressed`

즉 바이오 엔진의 최종 판단값인 `bio_arousal`을 멀티모달 내부 감정 점수로 변환하는 구조다.

8. 현재 확인된 상태

현재 로그 기준으로 확인된 것은 다음과 같다.

- 바이오 엔진이 payload를 생성하고 있음
- queue에 payload를 넣고 있음
- `run_bio_to_multimodal_queue.py`가 payload를 읽고 있음
- `RealSensorSource.update_window(payload)`가 실행되고 있음
- `get_latest_result()`에서 `SensorResult`가 생성되고 있음

즉 구조적으로는 아래 흐름이 동작 중이다.

BIO Engine -> Queue -> RealSensorSource

9. 현재 값이 0 또는 None으로 나오는 이유

초기 실행 로그에서 아래와 같은 값이 나올 수 있다.

- `bpm = None`
- `rmssd_use = None`
- `gsr_cur = None`
- `ppgQ = 0.0`
- `bio_arousal = 0.0`
- `gsr_fresh = False`
- `note = hrv_warming_up(need~30s)`

이 경우는 전달 실패가 아니라 센서값이 아직 충분히 쌓이지 않은 상태다.

예를 들면,

- HRV는 최소 일정 시간 이상 RR interval이 쌓여야 계산 가능
- GSR는 시리얼 값이 아직 최신으로 들어오지 않았을 수 있음
- PPG quality가 아직 낮을 수 있음

즉 이러한 상태는 다음처럼 해석해야 한다.

- 전달 구조 성공
- 초기 센서 데이터 품질은 아직 준비 중

10. 실행 파일

현재 BIO -> 멀티모달 전달 확인용 실행 파일은 아래다.

- `bio_engine/run_bio_to_multimodal_queue.py`

이 파일의 역할은 다음과 같다.

- `bio_engine_core.main(payload_queue=...)` 실행
- queue에서 최신 payload를 가져옴
- `RealSensorSource.update_window(payload)` 호출
- `get_latest_result()` 출력

현재 이 파일은 FER, 최종 fusion, OpenCV 웹캠 처리 등을 포함하지 않는다.

11. 현재 단계에서 하지 않은 것

현재 단계에서는 아래 작업은 포함하지 않았다.

- FER와의 최종 융합 실행
- `MultiModalEngine.step()`까지 자동 실행
- 최종 JSON 출력
- WebSocket 송신

이 단계의 목적은 어디까지나 바이오 엔진 결과를 멀티모달 입력부에 정상 전달하는지 확인하는 것이다.

12. 남아 있는 정리 포인트

현재 코드상 추가로 정리할 수 있는 부분은 아래 정도다.

- `safe/stressed` 와 `calm/stressed` 중 어느 표현을 최종 기준으로 할지 통일
- `hrv` 라는 이름에 `rmssd_use`를 넣는 것이 팀 내부 기준에 맞는지 확인
- `note`를 `raw_metrics`에 넣을지 별도 관리할지 결정
- 멀티모달 최종 엔진과 연결 시 실제 기대 키를 다시 한번 통일

13. 요약

현재 구조는 다음과 같다.

- 바이오 엔진이 실시간으로 센서값을 읽고 특징을 계산함
- 최종 payload를 생성함
- queue로 payload를 전달함
- 멀티모달 입력 객체가 그 payload를 받아 `SensorResult`로 변환함

현재까지는 BIO -> 멀티모달 입력 전달까지 확인된 상태다.
최종 멀티모달 융합 단계는 아직 따로 진행해야 한다.
