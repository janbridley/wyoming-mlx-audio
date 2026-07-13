"""Tests for the event handler."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from wyoming.asr import Transcribe
from wyoming.audio import AudioChunk, AudioStop
from wyoming.event import Event
from wyoming.info import Describe, Info

from wyoming_mlx_audio.handler import MlxAudioEventHandler, _pcm_to_float


class TestPcmToFloat:
    """Tests for _pcm_to_float function."""

    def test_silence(self) -> None:
        """Test conversion of silence (zeros)."""
        audio_bytes = bytes(100)  # 50 samples of silence (16-bit = 2 bytes)
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 50
        assert np.allclose(result, 0.0)

    def test_max_positive(self) -> None:
        """Test conversion of maximum positive value."""
        # int16 max is 32767
        audio_bytes = np.array([32767], dtype=np.int16).tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert np.isclose(result[0], 32767 / 32768.0)

    def test_max_negative(self) -> None:
        """Test conversion of maximum negative value."""
        # int16 min is -32768
        audio_bytes = np.array([-32768], dtype=np.int16).tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.dtype == np.float32
        assert len(result) == 1
        assert np.isclose(result[0], -1.0)

    def test_normalization_range(self) -> None:
        """Test that output is normalized to [-1, 1]."""
        # Create random int16 samples
        rng = np.random.default_rng(42)
        samples = rng.integers(-32768, 32767, size=1000, dtype=np.int16)
        audio_bytes = samples.tobytes()
        result = _pcm_to_float(audio_bytes)
        assert result.min() >= -1.0
        assert result.max() <= 1.0


def _make_model(text: str = "Hello world") -> MagicMock:
    """Create a mock model whose generate() returns an object with .text."""
    model = MagicMock()
    model.generate.return_value.text = text
    return model


class TestMlxAudioEventHandler:
    """Tests for MlxAudioEventHandler class."""

    @pytest.fixture
    def mock_wyoming_info(self) -> Info:
        """Create a mock Wyoming info object."""
        return MagicMock(spec=Info)

    @pytest.fixture
    def handler(
        self,
        mock_wyoming_info: Info,
    ) -> MlxAudioEventHandler:
        """Create a handler instance for testing."""
        handler = MlxAudioEventHandler(
            mock_wyoming_info,
            _make_model(),
            reader=MagicMock(),
            writer=MagicMock(),
        )
        handler.write_event = AsyncMock()
        return handler

    def test_init(self, mock_wyoming_info: Info) -> None:
        """Test handler initialization."""
        model = _make_model()
        handler = MlxAudioEventHandler(
            mock_wyoming_info,
            model,
            reader=MagicMock(),
            writer=MagicMock(),
        )
        assert handler._model is model
        assert handler._audio == b""

    def test_reset(self, handler: MlxAudioEventHandler) -> None:
        """Test audio buffer reset."""
        handler._audio = b"some audio data"
        handler._reset()
        assert handler._audio == b""

    @pytest.mark.asyncio
    async def test_handle_audio_chunk(self, handler: MlxAudioEventHandler) -> None:
        """Test handling of audio chunks."""
        chunk = AudioChunk(
            rate=16000,
            width=2,
            channels=1,
            audio=b"\x00\x00" * 100,
        )
        event = chunk.event()

        result = await handler.handle_event(event)

        assert result is True
        assert len(handler._audio) > 0

    @pytest.mark.asyncio
    async def test_handle_audio_stop(self, handler: MlxAudioEventHandler) -> None:
        """Test handling of audio stop event triggers transcription."""
        handler._audio = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second

        event = AudioStop().event()
        result = await handler.handle_event(event)

        assert result is False
        assert handler._audio == b""  # Should be reset
        handler._model.generate.assert_called_once()
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_describe(self, handler: MlxAudioEventHandler) -> None:
        """Test handling of describe event."""
        event = Describe().event()
        result = await handler.handle_event(event)

        assert result is True
        handler.write_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_passes_array_to_model(
        self,
        mock_wyoming_info: Info,
    ) -> None:
        """_transcribe hands the audio array directly to model.generate."""
        model = _make_model("custom result")
        handler = MlxAudioEventHandler(
            mock_wyoming_info,
            model,
            reader=MagicMock(),
            writer=MagicMock(),
        )
        audio = np.zeros(16000, dtype=np.float32)

        result = await handler._transcribe(audio)

        assert result == "custom result"
        model.generate.assert_called_once_with(audio)

    @pytest.mark.asyncio
    async def test_handle_transcribe_event_accepted(
        self,
        handler: MlxAudioEventHandler,
    ) -> None:
        """A Transcribe event is accepted (single-model: name is ignored)."""
        event = Transcribe(name="anything").event()
        result = await handler.handle_event(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_transcribe_event_without_context(
        self,
        handler: MlxAudioEventHandler,
    ) -> None:
        """A Transcribe event without context is accepted."""
        event = Transcribe(context=None).event()
        result = await handler.handle_event(event)

        assert result is True

    @pytest.mark.asyncio
    async def test_handle_unknown_event(self, handler: MlxAudioEventHandler) -> None:
        """Test handling of unknown event type."""
        event = Event(type="unknown-event-type")
        result = await handler.handle_event(event)

        # Should return True (continue processing)
        assert result is True
