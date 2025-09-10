from __future__ import annotations

import base64
import json
import hmac
import hashlib
import time
from typing import Dict, Any

# Secret key used for signing tokens
SECRET_KEY = "SECRET_KEY"
ALGORITHM = "HS256"  # Only HS256 supported here.
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # Default expiration: 24 hours.

# --- Password hashing / verification ---

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    """TODO: Salt Password"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against a hash"""
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == hashed

# --- Base64 helpers (URL-safe, no padding) ---

def _base64url_encode(data: bytes) -> str:
    """Encode bytes to base64url string"""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")

def _base64url_decode(data: str) -> bytes:
    """Decode a base64url string into bytes"""
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)

# --- Token creation ---

def create_access_token(sub: str, expires_delta_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """
    Create a signed JWT-like token
    - sub: subject (e.g., email)
    - exp: expiration time (in UNIX timestamp)
    """
    header = {"alg": ALGORITHM, "typ": "JWT"}
    payload: Dict[str, Any] = {"sub": sub, "exp": int(time.time()) + expires_delta_minutes * 60}

    # Encode header & payload
    header_b64 = _base64url_encode(json.dumps(header, separators=(",", ":"), sort_keys=True).encode())
    payload_b64 = _base64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())

    # Sign message
    message = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"

# --- Token decoding / validation ---

class InvalidTokenError(Exception):
    """Raised when a token is invalid"""
    pass

def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate token, returning payload"""
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
    except ValueError:
        raise InvalidTokenError("Token structure invalid")

    # Verify HMAC signature
    message = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest()
    if not hmac.compare_digest(expected_sig, _base64url_decode(signature_b64)):
        raise InvalidTokenError("Invalid signature")

    # Decode payload
    try:
        payload = json.loads(_base64url_decode(payload_b64).decode())
    except Exception:
        raise InvalidTokenError("Invalid payload encoding")

    # Expiration check
    if "exp" in payload and int(time.time()) > int(payload["exp"]):
        raise InvalidTokenError("Token expired")

    return payload
