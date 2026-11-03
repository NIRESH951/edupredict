"""
Password hashing with PBKDF2-HMAC-SHA256 (Python standard library only).

Stored format:  pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>
Plain-text passwords are never stored or logged.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import string

import config

_PREFIX = f"pbkdf2_{config.PBKDF2_ALGORITHM}"


def hash_password(password: str, *, iterations: int | None = None, salt: bytes | None = None) -> str:
    if not isinstance(password, str) or password == "":
        raise ValueError("Password must be a non-empty string")
    iterations = iterations or config.PBKDF2_ITERATIONS
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(config.PBKDF2_ALGORITHM, password.encode("utf-8"), salt, iterations)
    return "$".join([
        _PREFIX,
        str(iterations),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    ])


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time comparison of a candidate password with a stored hash."""
    try:
        prefix, iterations, salt_b64, hash_b64 = stored_hash.split("$")
        if prefix != _PREFIX:
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        candidate = hashlib.pbkdf2_hmac(
            config.PBKDF2_ALGORITHM, password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(candidate, expected)
    except (ValueError, TypeError, AttributeError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """True if the hash was made with fewer iterations than currently configured."""
    try:
        return int(stored_hash.split("$")[1]) < config.PBKDF2_ITERATIONS
    except (IndexError, ValueError):
        return True


def password_problems(password: str) -> list[str]:
    """Return a list of unmet password-policy rules (empty list = strong enough)."""
    problems = []
    if len(password) < config.PASSWORD_MIN_LENGTH:
        problems.append(f"at least {config.PASSWORD_MIN_LENGTH} characters")
    if not re.search(r"[A-Z]", password):
        problems.append("an uppercase letter")
    if not re.search(r"[a-z]", password):
        problems.append("a lowercase letter")
    if not re.search(r"\d", password):
        problems.append("a digit")
    if not re.search(r"[^A-Za-z0-9]", password):
        problems.append("a special character (e.g. @, #, !)")
    return problems


def generate_temporary_password(length: int = 12) -> str:
    """Generate a random password that satisfies the password policy."""
    alphabet = string.ascii_letters + string.digits
    while True:
        core = "".join(secrets.choice(alphabet) for _ in range(length - 2))
        pwd = core + secrets.choice("@#!$%") + secrets.choice(string.digits)
        if not password_problems(pwd):
            return pwd
