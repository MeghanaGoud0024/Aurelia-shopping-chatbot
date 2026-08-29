"""Application configuration.

All settings are environment-driven with production-safe defaults so that the
application boots on a clean machine with nothing but `pip install -r
requirements.txt`. Values are read once at import time and exposed through a
cached singleton, which keeps configuration access cheap inside request paths.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
WEB_DIR = PROJECT_ROOT / "web"


class Settings(BaseSettings):
    """Typed application settings, loaded from environment or `.env`."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="AURELIA_",
        extra="ignore",
    )

    # ----- Application ---------------------------------------------------
    app_name: str = "Aurelia AI Shopping Assistant"
    environment: str = "development"
    log_level: str = "INFO"
    seed: int = 20260830

    # ----- Persistence ---------------------------------------------------
    database_url: str = f"sqlite:///{DATA_DIR / 'aurelia.db'}"

    # ----- LLM provider --------------------------------------------------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.15
    llm_max_tokens: int = 1600
    #: Groq exposes reasoning effort on the gpt-oss family. "low" cuts reasoning
    #: tokens roughly fourfold with no measurable loss on tool selection, which
    #: is the only judgement call this agent asks the model to make. Raise it if
    #: you extend the assistant into genuinely multi-step planning.
    llm_reasoning_effort: str = "low"
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # ----- Guardrails ----------------------------------------------------
    guard_enabled: bool = True
    guard_injection_model: str = "meta-llama/llama-prompt-guard-2-86m"
    guard_injection_threshold: float = 0.85

    # ----- Agent loop ----------------------------------------------------
    max_tool_iterations: int = 6
    max_tool_calls_per_turn: int = 10

    # ----- Rate limiting -------------------------------------------------
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    @field_validator("llm_temperature")
    @classmethod
    def _validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 2.0:
            raise ValueError("llm_temperature must be between 0.0 and 2.0")
        return value

    @field_validator("guard_injection_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("guard_injection_threshold must be between 0.0 and 1.0")
        return value

    @property
    def llm_configured(self) -> bool:
        """True when a live LLM call is possible.

        When False the application still runs: the orchestrator falls back to a
        deterministic rule-based planner so the product surface is demonstrable
        offline. See `app/agent/fallback.py`.
        """
        return bool(self.llm_api_key.strip())

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
