# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Real-time multimodal emotion recognition system for edge devices. Combines:
- **FER** (Facial Expression Recognition): MediaPipe FaceMesh + TFLite model → 4 emotions (neutral, happy, sad, angry) + drowsiness
- **BIO** (Biometric signals): MAX30102 PPG sensor (BPM, HRV/RMSSD) + serial GSR sensor → arousal/stress
- **Fusion**: Weighted combination (FER 40% + BIO 60%) → stress level + 3-tier alert (NORMAL/WARNING/DANGER)
- **Output**: WebSocket broadcast (ws://0.0.0.0:8765) + CSV logs + OpenCV display

## Running the System

```bash
# Standard run with v4l2 camera
python main.py --mode run --tflite fer/models/efficientface_4cls_finetuned_fp16_float16.tflite

# TCP camera source
python main.py --mode run --tflite <model_path> --camera_backend tcp --tcp_host 127.0.0.1 --tcp_port 9999

# Visualization mode
python main.py --mode visualize --tflite <model_path>

# Key options
--cam_index 0           # Camera device index
--cam_w 1280 --cam_h 720
--min_det_conf 0.5 --min_track_conf 0.5
--no_flip               # Disable horizontal flip
```

## Dependencies (inferred from imports)

```
opencv-python
numpy
mediapipe
tflite-runtime  # or tensorflow
websockets
pyserial
smbus2          # for MAX30102 I2C
scipy
```

## Architecture

```
main.py (RuntimeController)
├── FER pipeline (main thread)
│   └── FERCore → MediaPipe + TFLite → FERQueuePublisher → queue
├── BIO pipeline (background thread)
│   └── bio_engine_core → MAX30102/GSR → payload_queue
└── Fusion pipeline (main loop)
    ├── RealFerSource (drains FER queue)
    ├── RealSensorSource (updates from BIO payloads)
    ├── MultiModalEngine.step() → FusionResult
    ├── WebSocket broadcast (JSON)
    └── CSV logging
```

### Module Responsibilities

- **`fer/fer_core.py`** — Face alignment via eye landmarks, ROI extraction (224×224), TFLite inference, EMA smoothing (α=0.65). Normalization: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].
- **`bio/bio_engine_core.py`** — Entry point `main(payload_queue)`. Runs at 100Hz PPG sampling, 1Hz output. Outputs: bpm, rmssd_use, gsr, arousal, confidence.
- **`bio/bio_ppg_processor.py`** — Peak detection, BPM with EMA, RMSSD calculation, RR interval validation, finger presence check (IR threshold).
- **`multimodal/multimodal_engine.py`** — Fusion weights: w_fer=0.4, w_sensor=0.6, ema_alpha=0.3. Alert thresholds: DANGER at stress≥0.75 or drowsy≥0.60, WARNING at stress≥0.50.
- **`multimodal/ws_server.py`** — AsyncIO WebSocket server running in background thread via `asyncio.run()`.

### Data Flow Between Modules

- FER → Fusion: `queue.Queue` of `FERPacket` objects (non-blocking, drops old if full)
- BIO → Fusion: `queue.Queue` of dict payloads `{bpm, rmssd_use, gsr, arousal, confidence}`
- Fusion → Clients: JSON via WebSocket

### Alert Logic

```python
# In multimodal_engine.py
if ema_stress >= 0.75 or drowsy >= 0.60:  → DANGER
elif ema_stress >= 0.50:                   → WARNING
else:                                      → NORMAL
```

### Fusion Formula

```python
# FER → stressed
stressed_fer = 0.6 * (sad + angry) + 0.4 * drowsy
safe_fer = 0.6 * (happy + neutral) + 0.4 * alert

# BIO → stressed
stressed_bio = bio_arousal * bio_conf
safe_bio = 1 - stressed_bio

# Weighted fusion
safe_fused = 0.4 * safe_fer + 0.6 * safe_bio
```

## Hardware Targets

- Linux edge device (ARM/x86) with camera
- MAX30102 pulse oximeter on I2C bus
- GSR sensor on serial (auto-detects /dev/ttyUSB0, /dev/ttyACM0, etc.)
- Designed for Raspberry Pi / Jetson class devices
