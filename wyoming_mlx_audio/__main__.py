#!/usr/bin/env python3
"""Wyoming server for MLX Audio (IBM Granite Speech)."""

import logging
from typing import Annotated

import typer

from . import __version__
from .const import DEFAULT_MODEL

_LOGGER = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print version and exit."""
    if value:
        typer.echo(__version__)
        raise typer.Exit


DEFAULT_URI = "tcp://0.0.0.0:7891"


@app.command()
def main(  # noqa: PLR0913
    uri: Annotated[
        str,
        typer.Option(envvar="MLX_AUDIO_URI", help="unix:// or tcp://"),
    ] = DEFAULT_URI,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            envvar="MLX_AUDIO_MODEL",
            help="HuggingFace repo of an mlx-audio STT model",
        ),
    ] = DEFAULT_MODEL,
    preload: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            envvar="MLX_AUDIO_PRELOAD",
            help="Load and warm up the model at startup",
        ),
    ] = False,
    download_only: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            "--download-only",
            envvar="MLX_AUDIO_DOWNLOAD_ONLY",
            help="Download the model and exit",
        ),
    ] = False,
    cache_dir: Annotated[
        str | None,
        typer.Option(
            envvar="MLX_AUDIO_CACHE_DIR",
            help="HuggingFace cache directory for model downloads",
        ),
    ] = None,
    debug: Annotated[  # noqa: FBT002
        bool,
        typer.Option(envvar="MLX_AUDIO_DEBUG", help="Log DEBUG messages"),
    ] = False,
    version: Annotated[  # noqa: ARG001, FBT002
        bool,
        typer.Option(
            "--version",
            callback=version_callback,
            is_eager=True,
            help="Print version and exit",
        ),
    ] = False,
) -> None:
    """Run the Wyoming MLX Audio (Granite Speech) server."""
    from rich.logging import RichHandler

    from .server import _load_model, run_server

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=debug)],
    )
    _LOGGER.debug(
        "model=%s, uri=%s, preload=%s, download_only=%s, debug=%s",
        model,
        uri,
        preload,
        download_only,
        debug,
    )

    if download_only:
        _LOGGER.info("Downloading model: %s", model)
        _load_model(model, cache_dir)
        _LOGGER.info("Done.")
        raise typer.Exit

    run_server(uri, model, debug=debug, preload=preload, cache_dir=cache_dir)


def run() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()
