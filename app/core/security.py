"""
API keys and credential encryption.

Keys are stored as SHA-256 hashes; the plaintext is shown once at creation.
Connector credentials are encrypted with Fernet derived from the app secret,
so a leaked database file does not hand over the customer's CRM.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from typing import Any

from app.core.config import settings

KEY_PREFIX = "lsk"


def generate_api_key() -> tuple[str, str, str]:
    """Return (plaintext, hash, display_prefix). Plaintext is never stored."""
    raw = secrets.token_urlsafe(32)
    plaintext = f"{KEY_PREFIX}_{raw}"
    return plaintext, hash_api_key(plaintext), plaintext[: len(KEY_PREFIX) + 9]


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_api_key(plaintext: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_api_key(plaintext), expected_hash)


def _fernet():
    """Build a Fernet from the app secret. Returns None if unavailable."""
    try:
        from cryptography.fernet import Fernet
    except ImportError:  # pragma: no cover - optional dependency
        return None

    digest = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secrets(payload: dict[str, Any] | None) -> str | None:
    """
    Encrypt a credential dict for storage.

    Without the cryptography package this falls back to base64 and marks the
    value plainly, so nobody mistakes obfuscation for encryption.
    """
    if not payload:
        return None

    serialized = json.dumps(payload).encode("utf-8")
    fernet = _fernet()
    if fernet is None:
        return "plain:" + base64.urlsafe_b64encode(serialized).decode("ascii")
    return "fernet:" + fernet.encrypt(serialized).decode("ascii")


def decrypt_secrets(stored: str | None) -> dict[str, Any]:
    if not stored:
        return {}

    scheme, _, body = stored.partition(":")
    if scheme == "plain":
        return json.loads(base64.urlsafe_b64decode(body.encode("ascii")))

    fernet = _fernet()
    if fernet is None:
        raise RuntimeError(
            "Stored credentials are encrypted but the cryptography package is missing. "
            "Install it with: pip install cryptography"
        )
    return json.loads(fernet.decrypt(body.encode("ascii")))


# Fields never written to logs, prediction records, or model features.
PII_FIELDS = frozenset(
    {"display_name", "email", "phone", "first_name", "last_name", "full_name", "address"}
)


def strip_pii(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove personal fields before a payload is logged or stored on a prediction."""
    return {k: v for k, v in payload.items() if k.lower() not in PII_FIELDS}
