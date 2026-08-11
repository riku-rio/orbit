from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SETTINGS_DIR = Path.home() / ".orbit"
SETTINGS_PATH = SETTINGS_DIR / "settings.json"


@dataclass
class Settings:
    model: str | None = None
    context_size: int | None = None
    temperature: float | None = None

    def ollama_options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {}
        if self.context_size is not None:
            options["num_ctx"] = self.context_size
        if self.temperature is not None:
            options["temperature"] = self.temperature
        return options

    def ollama_load_options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {}
        if self.context_size is not None:
            options["num_ctx"] = self.context_size
        return options


def load_settings() -> Settings:
    if not SETTINGS_PATH.exists():
        return Settings()

    try:
        data: Any = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()

    if not isinstance(data, dict):
        return Settings()

    model = data.get("model")
    context_size = data.get("context_size")
    temperature = data.get("temperature")

    return Settings(
        model=model if isinstance(model, str) and model else None,
        context_size=context_size if isinstance(context_size, int) and context_size > 0 else None,
        temperature=(
            float(temperature)
            if isinstance(temperature, (int, float)) and not isinstance(temperature, bool) and temperature >= 0
            else None
        ),
    )


def save_settings(settings: Settings) -> None:
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(settings), indent=2) + "\n",
        encoding="utf-8",
    )
