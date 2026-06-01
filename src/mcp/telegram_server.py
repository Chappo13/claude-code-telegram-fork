"""MCP server exposing Telegram-specific tools to Claude.

Runs as a stdio transport server. Each tool validates input and returns a
confirmation string. Actual Telegram delivery is handled by the bot's stream
callback which intercepts these tool calls.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".oga", ".opus", ".flac"}

mcp = FastMCP("telegram")


@mcp.tool()
async def send_image_to_user(file_path: str, caption: str = "") -> str:
    """Send an image file to the Telegram user.

    Args:
        file_path: Absolute path to the image file.
        caption: Optional caption to display with the image.

    Returns:
        Confirmation string when the image is queued for delivery.
    """
    path = Path(file_path)

    if not path.is_absolute():
        return f"Error: path must be absolute, got '{file_path}'"

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return (
            f"Error: unsupported image extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if not path.is_file():
        return f"Error: file not found: {file_path}"

    return f"Image queued for delivery: {path.name}"


@mcp.tool()
async def send_file_to_user(file_path: str, caption: str = "") -> str:
    """Send any file (document) to the Telegram user.

    Use this for scripts, transcripts, .md/.txt/.docx/.pdf, archives, etc.
    Telegram limit: 50 MB per file.

    Args:
        file_path: Absolute path to the file.
        caption: Optional caption shown under the document.

    Returns:
        Confirmation string when the file is queued for delivery.
    """
    path = Path(file_path)

    if not path.is_absolute():
        return f"Error: path must be absolute, got '{file_path}'"

    if not path.is_file():
        return f"Error: file not found: {file_path}"

    size = path.stat().st_size
    if size > 50 * 1024 * 1024:
        return f"Error: file exceeds 50 MB Telegram limit ({size} bytes)"

    return f"File queued for delivery: {path.name}"


@mcp.tool()
async def send_audio_to_user(
    file_path: str,
    caption: str = "",
    as_voice: bool = False,
) -> str:
    """Send an audio file to the Telegram user.

    Use this for TTS narration, podcast snippets, transcribed audio playback,
    etc. Voice-note mode requires .ogg/.opus; other formats fall back to
    regular audio attachments.

    Args:
        file_path: Absolute path to the audio file.
        caption: Optional caption shown with the audio.
        as_voice: If True and file is .ogg/.opus, sent as a voice note bubble.

    Returns:
        Confirmation string when the audio is queued for delivery.
    """
    path = Path(file_path)

    if not path.is_absolute():
        return f"Error: path must be absolute, got '{file_path}'"

    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        return (
            f"Error: unsupported audio extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    if not path.is_file():
        return f"Error: file not found: {file_path}"

    size = path.stat().st_size
    if size > 50 * 1024 * 1024:
        return f"Error: audio exceeds 50 MB Telegram limit ({size} bytes)"

    return f"Audio queued for delivery: {path.name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
