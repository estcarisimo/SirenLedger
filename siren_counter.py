#!/usr/bin/env python3
"""
siren_counter.py  – Raspberry Pi daemon that logs how many times it hears a siren each day.

Usage
-----
$ python3 siren_counter.py
"""
import datetime as dt
import json
import os
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from tflite_runtime.interpreter import Interpreter

########################################################################
# 1. Configuration
########################################################################
MODEL_PATH   = "yamnet.tflite"          # download link below
LABELS_PATH  = "yamnet_label_list.txt"  # plain‑text list of the 521 classes
THRESH       = 0.20    # confidence above which we count a detection
SAMPLE_RATE  = 16000   # YAMNet expects 16 kHz mono
# YAMNet expects exactly 15 600 samples (0.975 s at 16 kHz) per inference window
SAMPLES_PER_WINDOW = 15600
WINDOW_SEC   = SAMPLES_PER_WINDOW / SAMPLE_RATE  # ≈0.975 s
LOGFILE      = "siren_daily_counts.csv"

########################################################################
# 2. Load model and find siren class indices
########################################################################
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

KEYWORDS = ("siren", "police car", "ambulance", "fire engine", "fire truck")
SIREN_IDS = [i for i, lab in enumerate(labels)
             if any(k in lab.lower() for k in KEYWORDS)]

print(f"Tracking labels: {[labels[i] for i in SIREN_IDS]}")

########################################################################
# 3. Audio capture thread
########################################################################
audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """Run in RT audio thread – push raw audio to the queue."""
    if status:
        print("Audio status:", status)
    audio_q.put(indata.copy())

stream = sd.InputStream(channels=1,
                        samplerate=SAMPLE_RATE,
                        blocksize=SAMPLES_PER_WINDOW,
                        callback=audio_callback)

########################################################################
# 4. Main detection loop
########################################################################
def detect_loop():
    day        = dt.date.today()
    day_count  = 0

    while True:
        chunk = audio_q.get()              # shape = (frames, 1)
        mono  = np.squeeze(chunk).astype(np.float32)

        # Ensure the tensor has exactly 15 600 samples as YAMNet requires
        if mono.shape[0] != SAMPLES_PER_WINDOW:
            if mono.shape[0] > SAMPLES_PER_WINDOW:
                mono = mono[:SAMPLES_PER_WINDOW]
            else:  # pad with zeros if microphone underruns
                mono = np.pad(mono, (0, SAMPLES_PER_WINDOW - mono.shape[0]), mode='constant')

        # YAMNet wants shape (n_samples,) float32 in –1..1; now always the correct length
        interpreter.set_tensor(input_details[0]['index'], mono)
        interpreter.invoke()
        scores = interpreter.get_tensor(output_details[0]['index'])[0]

        # Aggregate confidence over all siren‑related classes
        conf = np.max(scores[SIREN_IDS])
        if conf >= THRESH:
            day_count += 1
            print(f"[{dt.datetime.now()}] Siren detected! conf={conf:.2f}")

        # Rotate logs at midnight
        if dt.date.today() != day:
            with open(LOGFILE, "a") as f:
                f.write(f"{day.isoformat()},{day_count}\n")
            print(f"Logged {day_count} sirens on {day}")
            day, day_count = dt.date.today(), 0

########################################################################
# 5. Run
########################################################################
if __name__ == "__main__":
    try:
        os.makedirs(os.path.dirname(LOGFILE) or ".", exist_ok=True)
        with stream:
            print("Listening…  press Ctrl+C to stop.")
            detect_loop()
    except KeyboardInterrupt:
        print("Exiting gracefully.")