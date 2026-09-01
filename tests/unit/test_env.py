# -*- coding: utf-8 -*-
from pathlib import Path

from backend.env import overlay_llm_env_from_file


def test_overlay_llm_keys_prefer_dotenv_over_shell(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=from-file\n"
        "MODEL_TIER=dev\n"
        "DIET_EXPERT_PG_DSN=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")
    monkeypatch.setenv("MODEL_TIER", "prod")
    monkeypatch.setenv("DIET_EXPERT_PG_DSN", "from-shell")

    overlay_llm_env_from_file(env_file)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-file"
    assert os.environ["MODEL_TIER"] == "dev"
    # Non-LLM keys stay shell-first.
    assert os.environ["DIET_EXPERT_PG_DSN"] == "from-shell"


def test_overlay_skips_empty_llm_keys(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    overlay_llm_env_from_file(env_file)

    import os

    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"
