"""Tests for the server module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from wyoming_mlx_audio import __version__
from wyoming_mlx_audio.const import SUPPORTED_LANGUAGES
from wyoming_mlx_audio.server import _create_wyoming_info, run_server

MODEL = "mlx-community/granite-speech-4.1-2b-nar-mlx"


class TestCreateWyomingInfo:
    """Tests for _create_wyoming_info function."""

    def test_creates_info_with_model(self) -> None:
        """A single AsrModel is created for the given repo."""
        info = _create_wyoming_info(MODEL)

        assert info.asr is not None
        assert len(info.asr) == 1
        assert info.asr[0].name == "mlx-audio"
        assert len(info.asr[0].models) == 1
        assert info.asr[0].models[0].name == MODEL

    def test_includes_supported_languages(self) -> None:
        """The model advertises the supported languages."""
        info = _create_wyoming_info(MODEL)
        assert info.asr[0].models[0].languages == SUPPORTED_LANGUAGES

    def test_includes_version(self) -> None:
        """Version is included in info."""
        info = _create_wyoming_info(MODEL)

        assert info.asr[0].version == __version__
        assert info.asr[0].models[0].version == __version__

    def test_attribution(self) -> None:
        """Attribution points to mlx-audio and the model repo."""
        info = _create_wyoming_info(MODEL)

        # Program attribution
        assert info.asr[0].attribution.name == "Blaizzy mlx-audio"
        assert "Blaizzy/mlx-audio" in info.asr[0].attribution.url

        # Model attribution
        assert info.asr[0].models[0].attribution.name == "mlx-community"
        assert MODEL in info.asr[0].models[0].attribution.url

    def test_installed_flags(self) -> None:
        """Installed flags are set."""
        info = _create_wyoming_info(MODEL)
        assert info.asr[0].installed is True
        assert info.asr[0].models[0].installed is True


class TestRunServer:
    """Tests for run_server function."""

    def test_logs_startup_banner(self) -> None:
        """run_server logs startup information including the model name."""
        with (
            patch("wyoming_mlx_audio.server._LOGGER") as mock_logger,
            patch("wyoming_mlx_audio.server._load_model", return_value=MagicMock()),
            patch("wyoming_mlx_audio.server.asyncio.run"),
        ):
            run_server(uri="tcp://localhost:7891", model="test-model", debug=False)

            calls = " ".join(str(call) for call in mock_logger.info.call_args_list)
            assert "Wyoming MLX Audio" in calls
            assert "test-model" in calls

    def test_runs_async_server(self) -> None:
        """run_server starts the async server with the debug flag."""
        with (
            patch("wyoming_mlx_audio.server._LOGGER"),
            patch("wyoming_mlx_audio.server._load_model", return_value=MagicMock()),
            patch("wyoming_mlx_audio.server.asyncio.run") as mock_run,
        ):
            run_server(uri="tcp://localhost:7891", model="test-model", debug=True)

            mock_run.assert_called_once()
            assert mock_run.call_args[1]["debug"] is True

    def test_handles_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt is handled gracefully."""
        with (
            patch("wyoming_mlx_audio.server._LOGGER"),
            patch("wyoming_mlx_audio.server._load_model", return_value=MagicMock()),
            patch(
                "wyoming_mlx_audio.server.asyncio.run",
                side_effect=KeyboardInterrupt,
            ),
        ):
            run_server(uri="tcp://localhost:7891", model="test-model", debug=False)

    def test_passes_loaded_model_to_handler_factory(self) -> None:
        """The handler factory receives the loaded model."""
        real_asyncio_run = asyncio.run
        mock_model = MagicMock()
        mock_server = MagicMock()
        mock_server.run = AsyncMock()

        with (
            patch("wyoming_mlx_audio.server._LOGGER"),
            patch(
                "wyoming_mlx_audio.server._load_model",
                return_value=mock_model,
            ),
            patch(
                "wyoming_mlx_audio.server.AsyncServer.from_uri",
                return_value=mock_server,
            ),
            patch("wyoming_mlx_audio.server.MlxAudioEventHandler") as mock_handler,
            patch(
                "wyoming_mlx_audio.server.asyncio.run",
                side_effect=lambda coro, debug: real_asyncio_run(coro, debug=debug),
            ),
        ):
            run_server(uri="tcp://localhost:7891", model="test-model", debug=False)

            mock_server.run.assert_awaited_once()
            await_args = mock_server.run.await_args
            assert await_args is not None
            handler_factory = await_args.args[0]
            reader = MagicMock()
            writer = MagicMock()
            handler_factory(reader, writer)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args
        assert call_args.args[1] is mock_model  # the loaded model
        assert call_args.args[2] is reader
        assert call_args.args[3] is writer

    def test_preload_warms_up_when_requested(self) -> None:
        """When preload is True the loaded model is warmed up."""
        real_asyncio_run = asyncio.run
        mock_model = MagicMock()
        mock_model.generate.return_value.text = "warmup"
        mock_server = MagicMock()
        mock_server.run = AsyncMock()

        with (
            patch("wyoming_mlx_audio.server._LOGGER"),
            patch(
                "wyoming_mlx_audio.server._load_model",
                return_value=mock_model,
            ),
            patch(
                "wyoming_mlx_audio.server.AsyncServer.from_uri",
                return_value=mock_server,
            ),
            patch(
                "wyoming_mlx_audio.server.asyncio.run",
                side_effect=lambda coro, debug: real_asyncio_run(coro, debug=debug),
            ),
        ):
            run_server(
                uri="tcp://localhost:7891",
                model="test-model",
                debug=False,
                preload=True,
            )

            mock_model.generate.assert_called_once()
