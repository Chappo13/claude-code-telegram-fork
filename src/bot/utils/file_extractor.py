"""Validate file/audio paths from MCP send_file_to_user / send_audio_to_user.

Mirrors image_extractor but accepts any file extension for documents,
and a curated audio set for send_audio_to_user. The stream callback in
orchestrator collects FileAttachment / AudioAttachment objects for later
Telegram delivery via reply_document / reply_audio.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()

AUDIO_EXTENSIONS = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
}

VOICE_EXTENSIONS = {".ogg", ".oga", ".opus"}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024
MAX_AUDIO_SIZE_BYTES = 50 * 1024 * 1024
MAX_FILES_PER_RESPONSE = 10


@dataclass
class FileAttachment:
    """A generic file to attach as a Telegram document."""

    path: Path
    original_reference: str


@dataclass
class AudioAttachment:
    """An audio file to attach via reply_audio or reply_voice."""

    path: Path
    mime_type: str
    original_reference: str
    as_voice: bool = False


def _resolve_within(file_path: str, approved_directory: Path) -> Optional[Path]:
    try:
        path = Path(file_path)
        if not path.is_absolute():
            return None
        resolved = path.resolve()
        try:
            resolved.relative_to(approved_directory.resolve())
        except ValueError:
            logger.debug(
                "MCP path outside approved directory",
                path=str(resolved),
                approved=str(approved_directory),
            )
            return None
        if not resolved.is_file():
            return None
        return resolved
    except (OSError, ValueError) as e:
        logger.debug("MCP path validation failed", path=file_path, error=str(e))
        return None


def validate_file_path(
    file_path: str,
    approved_directory: Path,
    caption: str = "",
) -> Optional[FileAttachment]:
    """Validate a path from send_file_to_user. Any extension allowed."""
    resolved = _resolve_within(file_path, approved_directory)
    if resolved is None:
        return None
    if resolved.stat().st_size > MAX_FILE_SIZE_BYTES:
        logger.debug(
            "MCP file too large", path=str(resolved), size=resolved.stat().st_size
        )
        return None
    return FileAttachment(path=resolved, original_reference=caption or file_path)


def validate_audio_path(
    file_path: str,
    approved_directory: Path,
    caption: str = "",
    as_voice: bool = False,
) -> Optional[AudioAttachment]:
    """Validate an audio path. Extension must be in AUDIO_EXTENSIONS."""
    resolved = _resolve_within(file_path, approved_directory)
    if resolved is None:
        return None
    if resolved.stat().st_size > MAX_AUDIO_SIZE_BYTES:
        return None
    ext = resolved.suffix.lower()
    mime_type = AUDIO_EXTENSIONS.get(ext)
    if not mime_type:
        return None
    if as_voice and ext not in VOICE_EXTENSIONS:
        as_voice = False
    return AudioAttachment(
        path=resolved,
        mime_type=mime_type,
        original_reference=caption or file_path,
        as_voice=as_voice,
    )
