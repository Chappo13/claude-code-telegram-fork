"""Read/write helpers for the bot's .env file.

The .env file sits at base-fork/.env (relative to this source tree).
Used by /env wizard to add new variables interactively without exposing
them in Telegram chat history.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

# orchestrator.py / env_manager.py live at base-fork/src/bot/
# Walk up to base-fork/.
ENV_FILE_PATH: Path = Path(__file__).resolve().parent.parent.parent / ".env"

# Variable names: uppercase letters, digits, underscore. Must start with letter
# or underscore. Conservative to prevent shell-injection style names.
VALID_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Max characters for a value (tokens are usually <500). Guards against accidental
# pasting of huge files.
MAX_VALUE_LEN = 4096

# Keys we refuse to modify from Telegram — too dangerous (auth/path overrides).
PROTECTED_KEYS = frozenset(
    {
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_BOT_USERNAME",
        "ALLOWED_USERS",
        "APPROVED_DIRECTORY",
        "DATABASE_URL",
    }
)


def _escape_env_value(value: str) -> str:
    """Wrap value in quotes if it has whitespace/special chars; escape inner quotes."""
    if value == "":
        return ""
    if any(c in value for c in ' \t\n\r#"\'$`\\'):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _unquote_env_value(value: str) -> str:
    """Inverse of _escape_env_value — handles common .env quoting."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        inner = value[1:-1]
        if value[0] == '"':
            inner = inner.replace('\\"', '"').replace("\\\\", "\\")
        return inner
    return value


def is_valid_key(name: str) -> bool:
    """Allowed key names. Used by wizard to validate user input."""
    if not name or len(name) > 64:
        return False
    if name in PROTECTED_KEYS:
        return False
    return bool(VALID_KEY_RE.match(name))


def mask_value(value: str) -> str:
    """Return a safe-to-show preview of a secret value."""
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}***{value[-4:]} ({len(value)} chars)"


def read_env_file(path: Path = ENV_FILE_PATH) -> Dict[str, str]:
    """Parse the .env file into a dict. Lines that aren't KEY=VAL are skipped."""
    if not path.exists():
        return {}
    result: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        if not VALID_KEY_RE.match(key):
            continue
        result[key] = _unquote_env_value(raw_value)
    return result


def list_user_keys(path: Path = ENV_FILE_PATH) -> List[Tuple[str, str]]:
    """Return list of (key, masked_value) sorted by key, excluding PROTECTED_KEYS."""
    env = read_env_file(path)
    out: List[Tuple[str, str]] = []
    for key in sorted(env.keys()):
        if key in PROTECTED_KEYS:
            continue
        out.append((key, mask_value(env[key])))
    return out


def write_env_var(name: str, value: str, path: Path = ENV_FILE_PATH) -> None:
    """Atomically update or append a KEY=VAL line in the .env file.

    Preserves existing comments and order. If the key exists, its line is
    rewritten in place. If not, the new line is appended at the end.
    """
    if not is_valid_key(name):
        raise ValueError(f"invalid key name: {name!r}")
    if len(value) > MAX_VALUE_LEN:
        raise ValueError(f"value too long ({len(value)} > {MAX_VALUE_LEN})")

    escaped = _escape_env_value(value)
    new_line = f"{name}={escaped}"

    if not path.exists():
        path.write_text(new_line + "\n", encoding="utf-8")
        path.chmod(0o600)
        return

    existing = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    replaced = False
    for line in existing:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key == name:
            out.append(new_line)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        # Trim trailing blanks, append new var, then a single trailing newline
        while out and not out[-1].strip():
            out.pop()
        out.append(new_line)
    # Atomic-ish: write to tmp then rename
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)
