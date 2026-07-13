"""Constants for Wyoming MLX Audio (Granite Speech)."""

# Default Granite Speech NAR (non-autoregressive) STT model.
DEFAULT_MODEL = "mlx-community/granite-speech-4.1-2b-nar-mlx"

# NAR requires exactly 16 kHz mono audio.
SAMPLE_RATE = 16000

# Languages advertised to Home Assistant as discovery metadata only.
# NOTE: the NAR model is multilingual and auto-detects the spoken language at
# inference time; it ignores any `language` argument. This list is purely for
# the Wyoming Describe event so HA can present a sensible language set.
SUPPORTED_LANGUAGES = [
    "en",
    "fr",
    "de",
    "es",
    "pt",
    "ja",
]
