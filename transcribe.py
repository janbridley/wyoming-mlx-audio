#!/usr/bin/env python3
"""Transcribe a WAV file through a running wyoming-mlx-audio server.

Resamples to 16 kHz mono on the client (so resampling can be timed), then sends
the audio and prints the transcript with resample / processing timings.
"""

import asyncio
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from wyoming.asr import Transcript, Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncClient

URI = "tcp://127.0.0.1:7891"
MODEL = "mlx-community/granite-speech-4.1-2b-nar-mlx"
TARGET_RATE = 16000


async def main(wav_path: str) -> None:
    samples, rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = len(samples) / rate

    t0 = time.perf_counter()
    if rate != TARGET_RATE:
        samples = resample_poly(samples, TARGET_RATE, rate)
    resample_ms = (time.perf_counter() - t0) * 1000
    duration_16k = len(samples) / TARGET_RATE

    # float32 [-1, 1] -> int16 PCM bytes
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    print(f"{duration:.1f}s ({rate} Hz) -> {URI}")

    async with AsyncClient.from_uri(URI) as client:
        await client.write_event(Transcribe(name=MODEL).event())
        await client.write_event(
            AudioChunk(rate=TARGET_RATE, width=2, channels=1, audio=pcm).event(),
        )
        await client.write_event(AudioStop().event())
        t1 = time.perf_counter()
        event = await asyncio.wait_for(client.read_event(), timeout=180)
        processing_s = time.perf_counter() - t1

    print(Transcript.from_event(event).text)
    print(
        f"resample: {resample_ms:.0f} ms | processing: {processing_s:.2f}s "
        f"| client RTFx: {duration_16k / processing_s:.1f}x"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <file.wav>")
    asyncio.run(main(sys.argv[1]))
