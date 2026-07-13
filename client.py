import asyncio
from wyoming.client import AsyncClient
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStop
import wave


async def main():
    with wave.open(
        "/Users/jenna/Downloads/sample-speech-1m.wav"
    ) as w:  # 16k, 16-bit, mono
        pcm = w.readframes(w.getnframes())
    async with AsyncClient.from_uri("tcp://127.0.0.1:7891") as c:
        await c.write_event(
            Transcribe(name="mlx-community/granite-speech-4.1-2b-nar-mlx").event()
        )
        await c.write_event(
            AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event()
        )
        await c.write_event(AudioStop().event())
        ev = await c.read_event()
        print(ev)  # Transcript


asyncio.run(main())
