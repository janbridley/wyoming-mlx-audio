"""Tests for the __main__ module."""

import re
from unittest.mock import patch

import pytest
import typer
from typer.testing import CliRunner

from wyoming_mlx_audio import __version__
from wyoming_mlx_audio.__main__ import DEFAULT_URI, app, version_callback
from wyoming_mlx_audio.const import DEFAULT_MODEL


class TestVersionCallback:
    """Tests for version_callback function."""

    def test_prints_version_and_exits(self) -> None:
        """Test that version callback prints version and exits."""
        with pytest.raises(typer.Exit):
            version_callback(value=True)

    def test_does_nothing_when_false(self) -> None:
        """Test that callback does nothing when value is False."""
        version_callback(value=False)


class TestCLI:
    """Tests for CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI test runner."""
        return CliRunner()

    def test_help_option(self, runner: CliRunner) -> None:
        """Test that --help works and shows the options."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        # Strip ANSI escape codes for reliable string matching
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
        assert "Wyoming MLX Audio" in output
        assert "--uri" in output
        assert "--model" in output
        assert "--download-only" in output

    def test_version_option(self, runner: CliRunner) -> None:
        """Test that --version works."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout

    def test_default_values(self) -> None:
        """Test that default values are correct."""
        assert DEFAULT_URI == "tcp://0.0.0.0:7891"
        assert DEFAULT_MODEL == "mlx-community/granite-speech-4.1-2b-nar-mlx"

    def test_main_calls_run_server(self, runner: CliRunner) -> None:
        """Test that main() calls run_server with uri and model."""
        with patch("wyoming_mlx_audio.server.run_server") as mock_run:
            runner.invoke(
                app,
                ["--uri", "tcp://localhost:9999", "--model", "test-model"],
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args.args[0] == "tcp://localhost:9999"  # uri
            assert call_args.args[1] == "test-model"  # model
            assert call_args.kwargs["debug"] is False

    def test_main_with_debug(self, runner: CliRunner) -> None:
        """Test that debug flag is passed correctly."""
        with patch("wyoming_mlx_audio.server.run_server") as mock_run:
            runner.invoke(app, ["--debug"])

            mock_run.assert_called_once()
            assert mock_run.call_args.kwargs["debug"] is True

    def test_main_with_env_vars(self, runner: CliRunner) -> None:
        """Test that environment variables are respected."""
        with patch("wyoming_mlx_audio.server.run_server") as mock_run:
            runner.invoke(
                app,
                [],
                env={
                    "MLX_AUDIO_URI": "tcp://env-host:8888",
                    "MLX_AUDIO_MODEL": "env-model",
                },
            )

            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args.args[0] == "tcp://env-host:8888"
            assert call_args.args[1] == "env-model"

    def test_download_only_exits_without_server(self, runner: CliRunner) -> None:
        """--download-only loads the model and exits without starting the server."""
        with (
            patch("wyoming_mlx_audio.server.run_server") as mock_run,
            patch("wyoming_mlx_audio.server._load_model") as mock_load,
        ):
            result = runner.invoke(app, ["--download-only", "--model", "test-model"])

            assert result.exit_code == 0
            mock_run.assert_not_called()
            mock_load.assert_called_once()
