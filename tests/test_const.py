"""Tests for the const module."""

from wyoming_mlx_audio.const import (
    DEFAULT_MODEL,
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
)


class TestConstants:
    """Tests for module constants."""

    def test_default_model_is_granite_nar(self) -> None:
        """The default model is the Granite Speech NAR MLX repo."""
        assert DEFAULT_MODEL == "mlx-community/granite-speech-4.1-2b-nar-mlx"

    def test_sample_rate_is_16k(self) -> None:
        """NAR requires exactly 16 kHz audio."""
        assert SAMPLE_RATE == 16000

    def test_languages_is_list(self) -> None:
        """SUPPORTED_LANGUAGES is a list."""
        assert isinstance(SUPPORTED_LANGUAGES, list)

    def test_contains_english(self) -> None:
        """English is advertised."""
        assert "en" in SUPPORTED_LANGUAGES

    def test_all_lowercase(self) -> None:
        """All language codes are lowercase."""
        for lang in SUPPORTED_LANGUAGES:
            assert lang == lang.lower(), f"Language code not lowercase: {lang}"

    def test_no_duplicates(self) -> None:
        """There are no duplicate language codes."""
        assert len(SUPPORTED_LANGUAGES) == len(set(SUPPORTED_LANGUAGES))

    def test_reasonable_count(self) -> None:
        """There is a sensible number of advertised languages."""
        assert 1 <= len(SUPPORTED_LANGUAGES) <= 20
