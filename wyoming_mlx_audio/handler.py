"""Event handler for clients of the server."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler

from .const import SAMPLE_RATE

if TYPE_CHECKING:
    from mlx.nn import Module
    from numpy.typing import NDArray
    from wyoming.event import Event

_LOGGER = logging.getLogger(__name__)

_WIDTH = 2
_CHANNELS = 1


def _pcm_to_float(audio_bytes: bytes) -> NDArray[np.float32]:
    """Convert 16-bit PCM audio bytes to float32 array normalized to [-1, 1]."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class MlxAudioEventHandler(AsyncEventHandler):
    """Event handler for clients."""

    def __init__(
        self,
        wyoming_info: Info,
        model: Module,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the event handler with a loaded mlx-audio model."""
        super().__init__(*args, **kwargs)
        self._model = model
        self._wyoming_info_event = wyoming_info.event()
        self._audio = b""
        self._audio_converter = AudioChunkConverter(
            rate=SAMPLE_RATE,
            width=_WIDTH,
            channels=_CHANNELS,
        )

    def _reset(self) -> None:
        """Reset the audio buffer."""
        self._audio = b""

    async def _transcribe(self, audio: NDArray[np.float32]) -> str:
        """Transcribe audio by passing the array straight into the model.

        mlx-audio's ``generate`` accepts a numpy ndarray, so the accumulated
        PCM is converted to float32 and handed over directly -- no temporary
        WAV file is written. Inference runs in a worker thread so the asyncio
        loop stays responsive while MLX blocks on the Metal device.
        """
        result = await asyncio.to_thread(self._model.generate, audio)
        return str(result.text)

    async def _handle_audio_chunk(self, event: Event) -> bool:
        """Handle incoming audio chunk."""
        if not self._audio:
            _LOGGER.debug("Receiving audio")
        chunk = AudioChunk.from_event(event)
        chunk = self._audio_converter.convert(chunk)
        self._audio += chunk.audio
        return True

    async def _handle_audio_stop(self) -> bool:
        """Handle end of audio stream and perform transcription."""
        _LOGGER.debug("Audio stopped, starting transcription")
        audio = _pcm_to_float(self._audio)
        duration = audio.shape[0] / SAMPLE_RATE
        start = time.perf_counter()
        text = await self._transcribe(audio)
        elapsed = time.perf_counter() - start
        rtfx = duration / elapsed if elapsed > 0 else 0.0
        _LOGGER.info(
            "Transcribed %.1fs in %.2fs (RTFx %.1fx): %s",
            duration,
            elapsed,
            rtfx,
            text,
        )
        await self.write_event(Transcript(text=text).event())
        _LOGGER.debug("Transcription sent")
        self._reset()
        return False

    async def _handle_describe(self) -> bool:
        """Handle describe request."""
        await self.write_event(self._wyoming_info_event)
        _LOGGER.debug("Sent info")
        return True

    async def handle_event(self, event: Event) -> bool:
        """Handle an event from the client."""
        if AudioChunk.is_type(event.type):
            return await self._handle_audio_chunk(event)

        if AudioStop.is_type(event.type):
            return await self._handle_audio_stop()

        if Transcribe.is_type(event.type):
            _LOGGER.debug("Transcribe event")
            return True

        if Describe.is_type(event.type):
            return await self._handle_describe()

        return True
