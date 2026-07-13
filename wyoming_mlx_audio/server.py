"""Wyoming server implementation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import TYPE_CHECKING, cast

import numpy as np
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .const import SAMPLE_RATE, SUPPORTED_LANGUAGES
from .handler import MlxAudioEventHandler

if TYPE_CHECKING:
    from mlx.nn import Module

_LOGGER = logging.getLogger(__name__)


def _create_wyoming_info(model: str) -> Info:
    """Create Wyoming service info for a single model."""
    return Info(
        asr=[
            AsrProgram(
                name="mlx-audio",
                description="IBM Granite Speech speech-to-text (MLX) for Apple Silicon",
                attribution=Attribution(
                    name="Blaizzy mlx-audio",
                    url="https://github.com/Blaizzy/mlx-audio",
                ),
                installed=True,
                version=__version__,
                models=[
                    AsrModel(
                        name=model,
                        description=model,
                        attribution=Attribution(
                            name="mlx-community",
                            url=f"https://huggingface.co/{model}",
                        ),
                        installed=True,
                        languages=SUPPORTED_LANGUAGES,
                        version=__version__,
                    ),
                ],
            ),
        ],
    )


def _load_model(model: str, cache_dir: str | None) -> Module:
    """Import mlx-audio (heavy) lazily and load the model (blocking)."""
    if cache_dir is not None:
        # Honour an explicit cache location for HuggingFace downloads.
        os.environ["HF_HOME"] = cache_dir
    from mlx_audio.stt import load_model

    return cast("Module", load_model(model))


async def _warm_up(model: Module) -> None:
    """Compile Metal kernels with ~1s of silence (non-fatal on failure)."""
    try:
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        await asyncio.to_thread(model.generate, silence)
        _LOGGER.info("Warm-up complete.")
    except Exception:  # noqa: BLE001
        _LOGGER.warning("Warm-up failed (non-fatal).", exc_info=True)


def run_server(
    uri: str,
    model: str,
    *,
    debug: bool,
    preload: bool = False,
    cache_dir: str | None = None,
) -> None:
    """Run the Wyoming MLX Audio (Granite Speech) server."""
    _LOGGER.info("🎤 Wyoming MLX Audio (Granite Speech)")
    _LOGGER.info("   URI:    %s", uri)
    _LOGGER.info("   Model:  %s", model)

    wyoming_info = _create_wyoming_info(model)

    async def _run() -> None:
        _LOGGER.info("📦 Loading model...")
        loaded = await asyncio.to_thread(_load_model, model, cache_dir)
        _LOGGER.info("✅ Model loaded!")
        if preload:
            _LOGGER.info("📦 Warming up...")
            await _warm_up(loaded)
        server = AsyncServer.from_uri(uri)
        _LOGGER.info("Ready")
        await server.run(
            lambda *args, **kwargs: MlxAudioEventHandler(
                wyoming_info,
                loaded,
                *args,
                **kwargs,
            ),
        )

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(), debug=debug)
