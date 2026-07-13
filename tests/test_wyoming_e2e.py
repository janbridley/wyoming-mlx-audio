"""End-to-end tests over a real Wyoming socket (transcription mocked).

These exercise the full client -> server -> handler -> model wire path with a
fake loaded model, so they run anywhere without a Metal GPU. The real mlx-audio
inference path is verified on the M1 via the dev clients ``client.py`` /
``transcribe.py`` (see the plan).
"""

import asyncio
import contextlib
import socket
from collections.abc import Awaitable, Callable
from pathlib import Path

import numpy as np
import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStop
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncServer

from wyoming_mlx_audio.const import DEFAULT_MODEL, SAMPLE_RATE, SUPPORTED_LANGUAGES
from wyoming_mlx_audio.handler import MlxAudioEventHandler
from wyoming_mlx_audio.server import _create_wyoming_info


class _FakeOutput:
    """Mimics mlx-audio's STTOutput."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    """Mimics a loaded mlx-audio STT model."""

    def __init__(self, text: str = "hello world") -> None:
        self._text = text

    def generate(self, audio: object, **kwargs: object) -> _FakeOutput:  # noqa: ARG002
        return _FakeOutput(self._text)


def _skip_if_no_unix_socket(socket_path: Path) -> None:
    """Skip the test if unix sockets cannot be bound in this environment."""
    with contextlib.suppress(FileNotFoundError):
        socket_path.unlink()
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(socket_path))
        sock.close()
    except OSError:
        pytest.skip("unix sockets unavailable in this environment")
    socket_path.unlink(missing_ok=True)


def _make_factory(wyoming_info: Info, model: _FakeModel) -> object:
    """Return a handler factory bound to the given model."""

    def factory(*args: object, **kwargs: object) -> MlxAudioEventHandler:
        return MlxAudioEventHandler(  # type: ignore[arg-type]
            wyoming_info,
            model,
            *args,
            **kwargs,
        )

    return factory


async def _run_against_server(
    socket_path: Path,
    model: _FakeModel,
    *,
    run_client: Callable[[AsyncClient], Awaitable[None]],
) -> None:
    """Start a server on a unix socket and drive it with ``run_client``."""
    uri = f"unix://{socket_path}"
    wyoming_info = _create_wyoming_info(DEFAULT_MODEL)
    server = AsyncServer.from_uri(uri)
    factory = _make_factory(wyoming_info, model)

    server_task = asyncio.create_task(_serve(server, factory))
    await asyncio.sleep(0.3)  # let the server bind
    try:
        async with AsyncClient.from_uri(uri) as client:
            await run_client(client)
    finally:
        server_task.cancel()
        await server_task
        socket_path.unlink(missing_ok=True)


async def _serve(server: AsyncServer, factory: object) -> None:
    """Run the server until cancelled."""
    with contextlib.suppress(asyncio.CancelledError):
        await server.run(factory)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_describe_and_transcribe(tmp_path: Path) -> None:
    """A client can describe and transcribe against the running server."""
    socket_path = tmp_path / "wyoming.sock"
    _skip_if_no_unix_socket(socket_path)
    model = _FakeModel("hello world")
    pcm = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()

    async def client_flow(client: AsyncClient) -> None:
        # Describe
        await client.write_event(Describe().event())
        event = await asyncio.wait_for(client.read_event(), timeout=5)
        info = Info.from_event(event)
        assert info.asr is not None
        assert info.asr[0].name == "mlx-audio"
        assert info.asr[0].models[0].name == DEFAULT_MODEL
        assert info.asr[0].models[0].languages == SUPPORTED_LANGUAGES

        # Transcribe
        await client.write_event(Transcribe(name=DEFAULT_MODEL).event())
        await client.write_event(
            AudioChunk(rate=SAMPLE_RATE, width=2, channels=1, audio=pcm).event(),
        )
        await client.write_event(AudioStop().event())
        event = await asyncio.wait_for(client.read_event(), timeout=5)
        transcript = Transcript.from_event(event)
        assert transcript.text == "hello world"

    await _run_against_server(
        socket_path,
        model,
        run_client=client_flow,
    )


@pytest.mark.asyncio
async def test_transcribe_without_named_model(tmp_path: Path) -> None:
    """A Transcribe event without (or with an unknown) name still transcribes.

    The server runs a single model, so the name is advisory only.
    """
    socket_path = tmp_path / "wyoming2.sock"
    _skip_if_no_unix_socket(socket_path)
    model = _FakeModel("ok")
    pcm = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()

    async def client_flow(client: AsyncClient) -> None:
        await client.write_event(Transcribe().event())
        await client.write_event(
            AudioChunk(rate=SAMPLE_RATE, width=2, channels=1, audio=pcm).event(),
        )
        await client.write_event(AudioStop().event())
        event = await asyncio.wait_for(client.read_event(), timeout=5)
        transcript = Transcript.from_event(event)
        assert transcript.text == "ok"

    await _run_against_server(
        socket_path,
        model,
        run_client=client_flow,
    )
