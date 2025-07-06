# Siren Counter – Raspberry Pi

A lightweight Python daemon that listens through a microphone on a Raspberry Pi and keeps a **daily count of emergency‑vehicle sirens** (police, ambulance, fire‑truck, civil‑defence).  
It leverages Google’s **[YAMNet](https://tfhub.dev/google/yamnet)** audio‑event classifier, so no training is required out‑of‑the‑box.

---

## Hardware requirements

| Component | Notes |
|-----------|-------|
| Raspberry Pi 3B +/ 4 (1 GB RAM +) | Any recent Pi works; tested on Pi OS (Bookworm) |
| Microphone | USB mic, Pi HAT, or I2S mic; must appear in `arecord ‑l` |
| Internet | Only needed once to download the model & labels |

---

## Quick‑start

### 1  Install system libraries

```bash
sudo apt update
sudo apt install -y python3-venv libportaudio2 portaudio19-dev wget
```

### 2  Create and activate a virtual‑environment

```bash
python3 -m venv ~/env-siren
source ~/env-siren/bin/activate
```

### 3  Install Python dependencies

> **You will provide an up‑to‑date `requirements.txt`** – e.g.  
> `numpy==1.26.4`, `sounddevice`, `tflite-runtime==2.14.0`, `numpy` pins, etc.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4  Download the YAMNet model & label list

```bash
cd /path/to/project

# Quantised TFLite model (~3.7 MB)
wget -O yamnet.tflite \
  "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite"

# 521‑class plain‑text label list (20 kB)
wget -O yamnet_label_list.txt \
  https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_label_list.txt
```

### 5  Run

```bash
python siren_counter.py
```

The console will print a line every time a siren is detected, and  
`**siren_daily_counts.csv**` will grow one line per day:

```text
2025-05-17,18
2025-05-18,21
```

Stop with **Ctrl +C**.

---

## Running on boot (optional)

Create `/etc/systemd/system/siren-counter.service`:

```ini
[Unit]
Description=Siren counter daemon
After=sound.target network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/noise
ExecStart=/home/pi/env-siren/bin/python /home/pi/noise/siren_counter.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now siren-counter
```

---

**Happy measuring, and may your Pi count sirens serenely!**
