import pytest

from core import app


def test_require_files_raises_on_missing(tmp_path):
    missing = tmp_path / "config.yaml"
    with pytest.raises(FileNotFoundError):
        app._require_files([str(missing)])


def test_validate_live_env_rejects_missing_env(monkeypatch):
    # Ensure all relevant env vars are absent
    keys = [
        "KRAKEN_API_KEY",
        "KRAKEN_API_SECRET",
        "BINANCE_API_KEY",
        "BINANCE_API_SECRET",
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "COINBASE_API_PASSPHRASE",
        "I_UNDERSTAND_LIVE_TRADING",
    ]
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError):
        app._validate_live_env("live")


def test_validate_live_env_noop_in_paper_mode(monkeypatch):
    # Paper mode should not require API keys.
    monkeypatch.delenv("KRAKEN_API_KEY", raising=False)
    app._validate_live_env("paper")

