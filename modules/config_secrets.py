"""Read config from environment and Streamlit secrets (Cloud-safe)."""

from __future__ import annotations

import os
from typing import Any, Optional


def _from_secrets(key: str, default: str = "") -> str:
    try:
        import streamlit as st

        secrets = getattr(st, "secrets", None)
        if secrets is None:
            return default
        if key in secrets:
            return str(secrets[key] or default)
        # nested [email] / [smtp] blocks
        for section in ("email", "smtp", "general"):
            try:
                block = secrets.get(section)  # type: ignore[attr-defined]
            except Exception:
                block = None
            if block is not None and key in block:
                return str(block[key] or default)
            # also try without prefix: EMAIL_TO inside [email] as "to"
            short = key.lower().removeprefix("email_").removeprefix("smtp_")
            if block is not None and short in block:
                return str(block[short] or default)
    except Exception:
        pass
    return default


def secret(key: str, default: str = "", *aliases: str) -> str:
    """Prefer env, then Streamlit secrets, then aliases."""
    val = os.getenv(key, "")
    if val:
        return str(val)
    val = _from_secrets(key, "")
    if val:
        return str(val)
    for alt in aliases:
        val = os.getenv(alt, "") or _from_secrets(alt, "")
        if val:
            return str(val)
    return default


def secret_bool(key: str, default: bool = False) -> bool:
    raw = secret(key, "true" if default else "false").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def get_email_to(default: str = "plant.manager@example.com") -> str:
    return secret(
        "EMAIL_TO",
        default,
        "SMTP_TO",
        "USER_EMAIL",
        "EMAIL_USER",
    )


def get_email_from(default: str = "oee-pulse@example.com") -> str:
    return secret("EMAIL_FROM", default, "SMTP_FROM")


def get_smtp_config() -> dict[str, Any]:
    return {
        "host": secret("SMTP_HOST", "", "EMAIL_SMTP_HOST"),
        "port": int(secret("SMTP_PORT", "587", "EMAIL_SMTP_PORT") or 587),
        "user": secret("SMTP_USER", "", "EMAIL_USER"),
        "password": secret("SMTP_PASSWORD", "", "EMAIL_PASSWORD"),
        "from_addr": get_email_from(),
        "to_addr": get_email_to(),
        "use_tls": secret_bool("SMTP_USE_TLS", True) or secret_bool("EMAIL_SMTP_USE_TLS", True),
    }


def get_gemini_api_key() -> str:
    return secret("GEMINI_API_KEY", "")


def get_openai_api_key() -> str:
    return secret("OPENAI_API_KEY", "")


def get_gemini_model() -> str:
    return secret("GEMINI_MODEL", "gemini-2.0-flash") or "gemini-2.0-flash"


def get_openai_model() -> str:
    return secret("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini"


def get_ai_default_provider() -> str:
    return (secret("AI_DEFAULT_PROVIDER", "gemini") or "gemini").lower()


def llm_status() -> dict[str, bool]:
    return {
        "gemini": bool(get_gemini_api_key()),
        "openai": bool(get_openai_api_key()),
        "offline": True,
    }


def demo_email_mode() -> bool:
    # Explicit demo flag wins; otherwise demo when SMTP host missing
    flagged = secret("OEE_PULSE_DEMO_MODE", "")
    if flagged:
        return flagged.strip().lower() in {"1", "true", "yes", "on"}
    return not bool(get_smtp_config()["host"])
