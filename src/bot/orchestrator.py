"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputMediaPhoto,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.sdk_integration import StreamUpdate
from ..config.settings import Settings
from ..projects import PrivateTopicsUnavailableError
from .env_manager import (
    is_valid_key,
    list_user_keys,
    write_env_var,
    PROTECTED_KEYS,
    MAX_VALUE_LEN,
)
from .utils.draft_streamer import DraftStreamer, generate_draft_id
from .utils.html_format import escape_html
from .utils.image_extractor import (
    ImageAttachment,
    should_send_as_photo,
    validate_image_path,
)

logger = structlog.get_logger()

_MEDIA_TYPE_MAP = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "webp": "image/webp",
}

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Tool name -> friendly emoji mapping for verbose output
_TOOL_ICONS: Dict[str, str] = {
    "Read": "\U0001f4d6",
    "Write": "\u270f\ufe0f",
    "Edit": "\u270f\ufe0f",
    "MultiEdit": "\u270f\ufe0f",
    "Bash": "\U0001f4bb",
    "Glob": "\U0001f50d",
    "Grep": "\U0001f50d",
    "LS": "\U0001f4c2",
    "Task": "\U0001f9e0",
    "TaskOutput": "\U0001f9e0",
    "WebFetch": "\U0001f310",
    "WebSearch": "\U0001f310",
    "NotebookRead": "\U0001f4d3",
    "NotebookEdit": "\U0001f4d3",
    "TodoRead": "\u2611\ufe0f",
    "TodoWrite": "\u2611\ufe0f",
}


def _tool_icon(name: str) -> str:
    """Return emoji for a tool, with a default wrench."""
    return _TOOL_ICONS.get(name, "\U0001f527")


@dataclass
class ActiveRequest:
    """Tracks an in-flight Claude request so it can be interrupted."""

    user_id: int
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    interrupted: bool = False
    progress_msg: Any = None  # telegram Message object


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps
        self._active_requests: Dict[int, ActiveRequest] = {}
        self._known_commands: frozenset[str] = frozenset()

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            context.user_data.pop("_thread_context", None)

            is_sync_bypass = handler.__name__ == "sync_threads"
            is_start_bypass = handler.__name__ in {"start_command", "agentic_start"}
            message_thread_id = self._extract_message_thread_id(update)
            should_enforce = self.settings.enable_project_threads

            if should_enforce:
                if self.settings.project_threads_mode == "private":
                    should_enforce = not is_sync_bypass and not (
                        is_start_bypass and message_thread_id is None
                    )
                else:
                    should_enforce = not is_sync_bypass

            if should_enforce:
                allowed = await self._apply_thread_routing_context(update, context)
                if not allowed:
                    return

            try:
                await handler(update, context)
            finally:
                if should_enforce:
                    self._persist_thread_state(context)

        return wrapped

    async def _apply_thread_routing_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> bool:
        """Enforce strict project-thread routing and load thread-local state."""
        manager = context.bot_data.get("project_threads_manager")
        if manager is None:
            await self._reject_for_thread_mode(
                update,
                "❌ <b>Project Thread Mode Misconfigured</b>\n\n"
                "Thread manager is not initialized.",
            )
            return False

        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return False

        if self.settings.project_threads_mode == "group":
            if chat.id != self.settings.project_threads_chat_id:
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False
        else:
            if getattr(chat, "type", "") != "private":
                await self._reject_for_thread_mode(
                    update,
                    manager.guidance_message(mode=self.settings.project_threads_mode),
                )
                return False

        message_thread_id = self._extract_message_thread_id(update)
        if not message_thread_id:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        project = await manager.resolve_project(chat.id, message_thread_id)
        if not project:
            await self._reject_for_thread_mode(
                update,
                manager.guidance_message(mode=self.settings.project_threads_mode),
            )
            return False

        state_key = f"{chat.id}:{message_thread_id}"
        thread_states = context.user_data.setdefault("thread_state", {})
        state = thread_states.get(state_key, {})

        project_root = project.absolute_path
        current_dir_raw = state.get("current_directory")
        current_dir = (
            Path(current_dir_raw).resolve() if current_dir_raw else project_root
        )
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        context.user_data["current_directory"] = current_dir
        context.user_data["claude_session_id"] = state.get("claude_session_id")
        context.user_data["_thread_context"] = {
            "chat_id": chat.id,
            "message_thread_id": message_thread_id,
            "state_key": state_key,
            "project_slug": project.slug,
            "project_root": str(project_root),
            "project_name": project.name,
        }
        return True

    def _persist_thread_state(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Persist compatibility keys back into per-thread state."""
        thread_context = context.user_data.get("_thread_context")
        if not thread_context:
            return

        project_root = Path(thread_context["project_root"])
        current_dir = context.user_data.get("current_directory", project_root)
        if not isinstance(current_dir, Path):
            current_dir = Path(str(current_dir))
        current_dir = current_dir.resolve()
        if not self._is_within(current_dir, project_root) or not current_dir.is_dir():
            current_dir = project_root

        thread_states = context.user_data.setdefault("thread_state", {})
        thread_states[thread_context["state_key"]] = {
            "current_directory": str(current_dir),
            "claude_session_id": context.user_data.get("claude_session_id"),
            "project_slug": thread_context["project_slug"],
        }

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        """Return True if path is within root."""
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_message_thread_id(update: Update) -> Optional[int]:
        """Extract topic/thread id from update message for forum/direct topics."""
        message = update.effective_message
        if not message:
            return None
        message_thread_id = getattr(message, "message_thread_id", None)
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            return message_thread_id
        dm_topic = getattr(message, "direct_messages_topic", None)
        topic_id = getattr(dm_topic, "topic_id", None) if dm_topic else None
        if isinstance(topic_id, int) and topic_id > 0:
            return topic_id
        # Telegram omits message_thread_id for the General topic in forum
        # supergroups; its canonical thread ID is 1.
        chat = update.effective_chat
        if chat and getattr(chat, "is_forum", False):
            return 1
        return None

    async def _reject_for_thread_mode(self, update: Update, message: str) -> None:
        """Send a guidance response when strict thread routing rejects an update."""
        query = update.callback_query
        if query:
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                await query.message.reply_text(message, parse_mode="HTML")
            return

        if update.effective_message:
            await update.effective_message.reply_text(message, parse_mode="HTML")

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register agentic handlers: commands + text/file/photo."""
        from .handlers import command

        # Commands
        handlers = [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("status", self.agentic_status),
            ("verbose", self.agentic_verbose),
            ("model", self.agentic_model),
            ("thinking", self.agentic_thinking),
            ("timeout", self.agentic_timeout),
            ("turns", self.agentic_turns),
            ("repo", self.agentic_repo),
            ("projects", self.agentic_projects),
            ("cost", self.agentic_cost),
            ("env", self.agentic_env),
            ("settings", self.agentic_settings),
            ("stop", self.agentic_stop),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))
        if self.settings.partner_mode:
            hidden = {"projects", "repo", "restart", "cost", "env", "status", "settings", "verbose", "model", "thinking", "timeout", "turns"}
            handlers = [(c, h) for c, h in handlers if c not in hidden]

        # Derive known commands dynamically — avoids drift when new commands are added
        self._known_commands: frozenset[str] = frozenset(cmd for cmd, _ in handlers)

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text),
            ),
            group=10,
        )

        # Unknown slash commands -> Claude (passthrough in agentic mode).
        # Registered commands are handled by CommandHandlers in group 0
        # (higher priority). This catches any /command not matched there
        # and forwards it to Claude, while skipping known commands to
        # avoid double-firing.
        app.add_handler(
            MessageHandler(
                filters.COMMAND,
                self._inject_deps(self._handle_unknown_command),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Voice messages -> transcribe -> Claude
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(self.agentic_voice)),
            group=10,
        )

        # Stop button callback (must be before cd: handler)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._handle_stop_callback),
                pattern=r"^stop:",
            )
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        # menu: callbacks (main menu navigation in /start)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_menu_callback),
                pattern=r"^menu:",
            )
        )

        # env: callbacks (env var wizard: add/cancel)
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_env_callback),
                pattern=r"^env:",
            )
        )

        # model: callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_model_callback),
                pattern=r"^model:",
            )
        )

        # thinking: callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_thinking_callback),
                pattern=r"^thinking:",
            )
        )

        # timeout: callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_timeout_callback),
                pattern=r"^timeout:",
            )
        )

        # turns: callbacks
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_turns_callback),
                pattern=r"^turns:",
            )
        )

        logger.info("Agentic handlers registered")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
            ("restart", command.restart_command),
        ]
        if self.settings.enable_project_threads:
            handlers.append(("sync_threads", command.sync_threads))

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.VOICE, self._inject_deps(message.handle_voice)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            commands = [
                BotCommand("start", "Menu"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("status", "Show session status"),
                BotCommand("projects", "List/switch projects"),
                BotCommand("cost", "Show today's spend"),
                BotCommand("env", "Manage .env variables (add tokens)"),
                BotCommand("settings", "Settings overview"),
                BotCommand("stop", "Stop the currently running task"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
                BotCommand("repo", "Alias for /projects"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            if self.settings.partner_mode:
                hidden = {"projects", "repo", "restart", "cost", "env", "status", "settings", "verbose", "model", "thinking"}
                commands = [c for c in commands if c.command not in hidden]
            return commands
        else:
            commands = [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
                BotCommand("restart", "Restart the bot"),
            ]
            if self.settings.enable_project_threads:
                commands.append(BotCommand("sync_threads", "Sync project topics"))
            return commands

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        sync_line = ""
        if (
            self.settings.enable_project_threads
            and self.settings.project_threads_mode == "private"
        ):
            if (
                not update.effective_chat
                or getattr(update.effective_chat, "type", "") != "private"
            ):
                await update.message.reply_text(
                    "🚫 <b>Private Topics Mode</b>\n\n"
                    "Use this bot in a private chat and run <code>/start</code> there.",
                    parse_mode="HTML",
                )
                return
            manager = context.bot_data.get("project_threads_manager")
            if manager:
                try:
                    result = await manager.sync_topics(
                        context.bot,
                        chat_id=update.effective_chat.id,
                    )
                    sync_line = (
                        "\n\n🧵 Topics synced"
                        f" (created {result.created}, reused {result.reused})."
                    )
                except PrivateTopicsUnavailableError:
                    await update.message.reply_text(
                        manager.private_topics_unavailable_message(),
                        parse_mode="HTML",
                    )
                    return
                except Exception:
                    sync_line = "\n\n🧵 Topic sync failed. Run /sync_threads to retry."
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        ro_badge = " · 🔒 read-only" if self.settings.read_only_mode else ""
        if self.settings.partner_mode:
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "🆕 Новая сессия", callback_data="menu:new"
                        ),
                    ],
                ]
            )
        else:
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "📁 Проекты", callback_data="menu:projects"
                        ),
                        InlineKeyboardButton(
                            "📊 Статус", callback_data="menu:status"
                        ),
                    ],
                    [
                        InlineKeyboardButton(
                            "💰 Расход", callback_data="menu:cost"
                        ),
                        InlineKeyboardButton("🔑 Ключи", callback_data="menu:env"),
                    ],
                    [
                        InlineKeyboardButton(
                            "⚙️ Настройки", callback_data="menu:settings"
                        ),
                        InlineKeyboardButton(
                            "🆕 Новая сессия", callback_data="menu:new"
                        ),
                    ],
                ]
            )

        welcome_text: Optional[str] = None
        wmf = self.settings.welcome_message_file
        if wmf is not None:
            try:
                raw = Path(wmf).read_text(encoding="utf-8")
                welcome_text = raw.format(
                    first_name=safe_name,
                    dir=dir_display,
                    ro_badge=ro_badge,
                    sync_line=sync_line,
                )
            except (OSError, KeyError, ValueError) as exc:
                logger.warning(
                    "welcome_message_file unreadable, falling back to default",
                    path=str(wmf),
                    error=str(exc),
                )

        if welcome_text is None:
            welcome_text = (
                f"Hi {safe_name}! I'm your AI coding assistant.{ro_badge}\n"
                f"Just tell me what you need — I can read, write, and run code.\n\n"
                f"Working in: {dir_display}\n"
                f"Commands: /new · /status · /projects · /cost · /verbose"
                f"{sync_line}"
            )

        await update.message.reply_text(
            welcome_text,
            parse_mode="HTML",
            reply_markup=keyboard,
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.user_data["claude_session_id"] = None
        context.user_data["session_started"] = True
        context.user_data["force_new_session"] = True

        await update.message.reply_text("Session reset. What's next?")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.user_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        await update.message.reply_text(
            f"📂 {dir_display} · Session: {session_status}{cost_str}"
        )

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.user_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (final response only)\n"
                "  1 = normal (tools + reasoning)\n"
                "  2 = detailed (tools with inputs + reasoning)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.user_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        activity_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build the progress message text based on activity so far."""
        if not activity_log:
            return "Working..."

        elapsed = time.time() - start_time
        lines: List[str] = [f"Working... ({elapsed:.0f}s)\n"]

        for entry in activity_log[-15:]:  # Show last 15 entries max
            kind = entry.get("kind", "tool")
            if kind == "text":
                # Claude's intermediate reasoning/commentary
                snippet = entry.get("detail", "")
                if verbose_level >= 2:
                    lines.append(f"\U0001f4ac {snippet}")
                else:
                    # Level 1: one short line
                    lines.append(f"\U0001f4ac {snippet[:80]}")
            else:
                # Tool call
                icon = _tool_icon(entry["name"])
                if verbose_level >= 2 and entry.get("detail"):
                    lines.append(f"{icon} {entry['name']}: {entry['detail']}")
                else:
                    lines.append(f"{icon} {entry['name']}")

        if len(activity_log) > 15:
            lines.insert(1, f"... ({len(activity_log) - 15} earlier entries)\n")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short summary of tool input for verbose level 2."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                # Show just the filename, not the full path
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return pattern[:60]
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:100])[:80]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:60]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:60]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:60]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task.

        Sends typing every *interval* seconds, independently of
        stream events. Cancel the returned task in a ``finally``
        block.
        """

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
        reply_markup: Optional[InlineKeyboardMarkup] = None,
        mcp_images: Optional[List[ImageAttachment]] = None,
        approved_directory: Optional[Path] = None,
        draft_streamer: Optional[DraftStreamer] = None,
        interrupt_event: Optional[asyncio.Event] = None,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback for verbose progress updates.

        When *mcp_images* is provided, the callback also intercepts
        ``send_image_to_user`` tool calls and collects validated
        :class:`ImageAttachment` objects for later Telegram delivery.

        When *draft_streamer* is provided, tool activity and assistant
        text are streamed to the user in real time via
        ``sendMessageDraft``.

        Returns None when verbose_level is 0 **and** no MCP image
        collection or draft streaming is requested.
        Typing indicators are handled by a separate heartbeat task.
        """
        need_mcp_intercept = mcp_images is not None and approved_directory is not None

        if verbose_level == 0 and not need_mcp_intercept and draft_streamer is None:
            return None

        last_edit_time = [0.0]  # mutable container for closure

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Stop all streaming activity after interrupt
            if interrupt_event is not None and interrupt_event.is_set():
                return

            # Intercept send_image_to_user MCP tool calls.
            # The SDK namespaces MCP tools as "mcp__<server>__<tool>",
            # so match both the bare name and the namespaced variant.
            if update_obj.tool_calls and need_mcp_intercept:
                for tc in update_obj.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name == "send_image_to_user" or tc_name.endswith(
                        "__send_image_to_user"
                    ):
                        tc_input = tc.get("input", {})
                        file_path = tc_input.get("file_path", "")
                        caption = tc_input.get("caption", "")
                        img = validate_image_path(
                            file_path, approved_directory, caption
                        )
                        if img:
                            mcp_images.append(img)

            # Capture tool calls
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    if verbose_level >= 1:
                        tool_log.append(
                            {"kind": "tool", "name": name, "detail": detail}
                        )
                    if draft_streamer:
                        icon = _tool_icon(name)
                        line = (
                            f"{icon} {name}: {detail}" if detail else f"{icon} {name}"
                        )
                        await draft_streamer.append_tool(line)

            # Capture assistant text (reasoning / commentary)
            if update_obj.type == "assistant" and update_obj.content:
                text = update_obj.content.strip()
                if text:
                    first_line = text.split("\n", 1)[0].strip()
                    if first_line:
                        if verbose_level >= 1:
                            tool_log.append(
                                {"kind": "text", "detail": first_line[:120]}
                            )
                        if draft_streamer:
                            await draft_streamer.append_tool(
                                f"\U0001f4ac {first_line[:120]}"
                            )

            # Stream text to user via draft (prefer token deltas;
            # skip full assistant messages to avoid double-appending)
            if draft_streamer and update_obj.content:
                if update_obj.type == "stream_delta":
                    await draft_streamer.append_text(update_obj.content)

            # Throttle progress message edits to avoid Telegram rate limits
            if not draft_streamer and verbose_level >= 1:
                now = time.time()
                if (now - last_edit_time[0]) >= 2.0 and tool_log:
                    last_edit_time[0] = now
                    new_text = self._format_verbose_progress(
                        tool_log, verbose_level, start_time
                    )
                    try:
                        await progress_msg.edit_text(
                            new_text, reply_markup=reply_markup
                        )
                    except Exception:
                        pass

        return _on_stream

    async def _send_images(
        self,
        update: Update,
        images: List[ImageAttachment],
        reply_to_message_id: Optional[int] = None,
        caption: Optional[str] = None,
        caption_parse_mode: Optional[str] = None,
    ) -> bool:
        """Send extracted images as a media group (album) or documents.

        If *caption* is provided and fits (≤1024 chars), it is attached to the
        photo / first album item so text + images appear as one message.

        Returns True if the caption was successfully embedded in the photo message.
        """
        photos: List[ImageAttachment] = []
        documents: List[ImageAttachment] = []
        for img in images:
            if should_send_as_photo(img.path):
                photos.append(img)
            else:
                documents.append(img)

        # Telegram caption limit
        use_caption = bool(
            caption and len(caption) <= 1024 and photos and not documents
        )
        caption_sent = False

        # Send raster photos as a single album (Telegram groups 2-10 items)
        if photos:
            try:
                if len(photos) == 1:
                    with open(photos[0].path, "rb") as f:
                        await update.message.reply_photo(
                            photo=f,
                            reply_to_message_id=reply_to_message_id,
                            caption=caption if use_caption else None,
                            parse_mode=caption_parse_mode if use_caption else None,
                        )
                    caption_sent = use_caption
                else:
                    media = []
                    file_handles = []
                    for idx, img in enumerate(photos[:10]):
                        fh = open(img.path, "rb")  # noqa: SIM115
                        file_handles.append(fh)
                        media.append(
                            InputMediaPhoto(
                                media=fh,
                                caption=caption if use_caption and idx == 0 else None,
                                parse_mode=(
                                    caption_parse_mode
                                    if use_caption and idx == 0
                                    else None
                                ),
                            )
                        )
                    try:
                        await update.message.chat.send_media_group(
                            media=media,
                            reply_to_message_id=reply_to_message_id,
                        )
                        caption_sent = use_caption
                    finally:
                        for fh in file_handles:
                            fh.close()
            except Exception as e:
                logger.warning("Failed to send photo album", error=str(e))

        # Send SVGs / large files as documents (one by one — can't mix in album)
        for img in documents:
            try:
                with open(img.path, "rb") as f:
                    await update.message.reply_document(
                        document=f,
                        filename=img.path.name,
                        reply_to_message_id=reply_to_message_id,
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send document image",
                    path=str(img.path),
                    error=str(e),
                )

        return caption_sent

    async def agentic_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Direct Claude passthrough. Simple progress. No suggestions."""
        user_id = update.effective_user.id
        message_text = update.message.text

        # Intercept env wizard input before Claude sees it.
        if context.user_data.get("env_wizard_step"):
            await self._handle_env_wizard_input(update, context, message_text)
            return

        logger.info(
            "Agentic text message",
            user_id=user_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"⏱️ {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)

        # Create Stop button and interrupt event
        interrupt_event = asyncio.Event()
        stop_kb = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Stop", callback_data=f"stop:{user_id}")]]
        )
        progress_msg = await update.message.reply_text(
            "Working...", reply_markup=stop_kb
        )

        # Register active request for stop callback
        active_request = ActiveRequest(
            user_id=user_id,
            interrupt_event=interrupt_event,
            progress_msg=progress_msg,
        )
        self._active_requests[user_id] = active_request

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            self._active_requests.pop(user_id, None)
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration.",
                reply_markup=None,
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        # --- Verbose progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        mcp_images: List[ImageAttachment] = []

        # Stream drafts (private chats only)
        draft_streamer: Optional[DraftStreamer] = None
        if self.settings.enable_stream_drafts and chat.type == "private":
            draft_streamer = DraftStreamer(
                bot=context.bot,
                chat_id=chat.id,
                draft_id=generate_draft_id(),
                message_thread_id=update.message.message_thread_id,
                throttle_interval=self.settings.stream_draft_interval,
            )

        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            start_time,
            reply_markup=stop_kb,
            mcp_images=mcp_images,
            approved_directory=self.settings.approved_directory,
            draft_streamer=draft_streamer,
            interrupt_event=interrupt_event,
        )

        # Independent typing heartbeat — stays alive even with no stream events
        heartbeat = self._start_typing_heartbeat(chat)

        success = True
        try:
            claude_response = await claude_integration.run_command(
                prompt=self._maybe_prefix_readonly(message_text),
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                interrupt_event=interrupt_event,
            )

            # New session created successfully — clear the one-shot flag
            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response (no reply_markup — strip keyboards)
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)

            response_content = claude_response.content
            if claude_response.interrupted:
                response_content = (
                    response_content or ""
                ) + "\n\n_(Interrupted by user)_"

            formatted_messages = formatter.format_claude_response(response_content)

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(e), parse_mode="HTML")
            ]
        finally:
            heartbeat.cancel()
            self._active_requests.pop(user_id, None)
            if draft_streamer:
                try:
                    await draft_streamer.flush()
                except Exception:
                    logger.debug("Draft flush failed in finally block", user_id=user_id)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls)
        images: List[ImageAttachment] = mcp_images

        # Try to combine text + images in one message when possible
        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        # Send text messages (skip if caption was already embedded in photos)
        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                try:
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,  # No keyboards in agentic mode
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)
                except Exception as send_err:
                    logger.warning(
                        "Failed to send HTML response, retrying as plain text",
                        error=str(send_err),
                        message_index=i,
                    )
                    try:
                        await update.message.reply_text(
                            message.text,
                            reply_markup=None,
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )
                    except Exception as plain_err:
                        await update.message.reply_text(
                            f"Failed to deliver response "
                            f"(Telegram error: {str(plain_err)[:150]}). "
                            f"Please try again.",
                            reply_to_message_id=(
                                update.message.message_id if i == 0 else None
                            ),
                        )

            # Send images separately if caption wasn't used
            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message.
        # Flag is only cleared after a successful run so retries keep the intent.
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_doc: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_doc,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
            )

            if force_new:
                context.user_data["force_new_session"] = False

            context.user_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            try:
                await progress_msg.delete()
            except Exception:
                logger.debug("Failed to delete progress message, ignoring")

            # Use MCP-collected images (from send_image_to_user tool calls)
            images: List[ImageAttachment] = mcp_images_doc

            caption_sent = False
            if images and len(formatted_messages) == 1:
                msg = formatted_messages[0]
                if msg.text and len(msg.text) <= 1024:
                    try:
                        caption_sent = await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                            caption=msg.text,
                            caption_parse_mode=msg.parse_mode,
                        )
                    except Exception as img_err:
                        logger.warning("Image+caption send failed", error=str(img_err))

            if not caption_sent:
                for i, message in enumerate(formatted_messages):
                    await update.message.reply_text(
                        message.text,
                        parse_mode=message.parse_mode,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                    if i < len(formatted_messages) - 1:
                        await asyncio.sleep(0.5)

                if images:
                    try:
                        await self._send_images(
                            update,
                            images,
                            reply_to_message_id=update.message.message_id,
                        )
                    except Exception as img_err:
                        logger.warning("Image send failed", error=str(img_err))

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )
            fmt = processed_image.metadata.get("format", "png")
            images = [
                {
                    "data": processed_image.base64_data,
                    "media_type": _MEDIA_TYPE_MAP.get(fmt, "image/png"),
                }
            ]

            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_image.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
                images=images,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def agentic_voice(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Transcribe voice message -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        voice_handler = features.get_voice_handler() if features else None

        if not voice_handler:
            await update.message.reply_text(self._voice_unavailable_message())
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Transcribing...")

        try:
            voice = update.message.voice
            processed_voice = await voice_handler.process_voice_message(
                voice, update.message.caption
            )

            await progress_msg.edit_text("Working...")
            await self._handle_agentic_media_message(
                update=update,
                context=context,
                prompt=processed_voice.prompt,
                progress_msg=progress_msg,
                user_id=user_id,
                chat=chat,
            )

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(_format_error_message(e), parse_mode="HTML")
            logger.error(
                "Claude voice processing failed", error=str(e), user_id=user_id
            )

    async def _handle_agentic_media_message(
        self,
        *,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        prompt: str,
        progress_msg: Any,
        user_id: int,
        chat: Any,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Run a media-derived prompt through Claude and send responses."""
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.user_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.user_data.get("claude_session_id")
        force_new = bool(context.user_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        mcp_images_media: List[ImageAttachment] = []
        on_stream = self._make_stream_callback(
            verbose_level,
            progress_msg,
            tool_log,
            time.time(),
            mcp_images=mcp_images_media,
            approved_directory=self.settings.approved_directory,
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                images=images,
            )
        finally:
            heartbeat.cancel()

        if force_new:
            context.user_data["force_new_session"] = False

        context.user_data["claude_session_id"] = claude_response.session_id

        from .handlers.message import _update_working_directory_from_claude_response

        _update_working_directory_from_claude_response(
            claude_response, context, self.settings, user_id
        )

        from .utils.formatting import ResponseFormatter

        formatter = ResponseFormatter(self.settings)
        formatted_messages = formatter.format_claude_response(claude_response.content)

        try:
            await progress_msg.delete()
        except Exception:
            logger.debug("Failed to delete progress message, ignoring")

        # Use MCP-collected images (from send_image_to_user tool calls).
        images: List[ImageAttachment] = mcp_images_media

        caption_sent = False
        if images and len(formatted_messages) == 1:
            msg = formatted_messages[0]
            if msg.text and len(msg.text) <= 1024:
                try:
                    caption_sent = await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                        caption=msg.text,
                        caption_parse_mode=msg.parse_mode,
                    )
                except Exception as img_err:
                    logger.warning("Image+caption send failed", error=str(img_err))

        if not caption_sent:
            for i, message in enumerate(formatted_messages):
                if not message.text or not message.text.strip():
                    continue
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)

            if images:
                try:
                    await self._send_images(
                        update,
                        images,
                        reply_to_message_id=update.message.message_id,
                    )
                except Exception as img_err:
                    logger.warning("Image send failed", error=str(img_err))

    async def _handle_unknown_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Forward unknown slash commands to Claude in agentic mode.

        Known commands are handled by their own CommandHandlers (group 0);
        this handler fires for *every* COMMAND message in group 10 but
        returns immediately when the command is registered, preventing
        double execution.
        """
        msg = update.effective_message
        if not msg or not msg.text:
            return
        cmd = msg.text.split()[0].lstrip("/").split("@")[0].lower()
        if cmd in self._known_commands:
            return  # let the registered CommandHandler take care of it
        # Forward unrecognised /commands to Claude as natural language
        await self.agentic_text(update, context)

    def _voice_unavailable_message(self) -> str:
        """Return provider-aware guidance when voice feature is unavailable."""
        if self.settings.voice_provider == "local":
            return (
                "Voice processing is not available. "
                "Ensure whisper.cpp is installed and the model file exists. "
                "Check WHISPER_CPP_BINARY_PATH and WHISPER_CPP_MODEL_PATH settings."
            )
        return (
            "Voice processing is not available. "
            f"Set {self.settings.voice_provider_api_key_env} "
            f"for {self.settings.voice_provider_display_name} and install "
            'voice extras with: pip install "claude-code-telegram[voice]"'
        )

    async def agentic_repo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """List repos in workspace or switch to one.

        /repo          — list subdirectories with git indicators
        /repo <name>   — switch to that directory, resume session if available
        """
        if context.user_data.get("_invoked_from_menu"):
            args: List[str] = []
        else:
            args = update.message.text.split()[1:] if update.message.text else []
        base = self.settings.approved_directory
        current_dir = context.user_data.get("current_directory", base)

        if args:
            # Switch to named repo
            target_name = args[0]
            target_path = base / target_name
            if not target_path.is_dir():
                await update.message.reply_text(
                    f"Directory not found: <code>{escape_html(target_name)}</code>",
                    parse_mode="HTML",
                )
                return

            context.user_data["current_directory"] = target_path

            # Try to find a resumable session
            claude_integration = context.bot_data.get("claude_integration")
            session_id = None
            if claude_integration:
                existing = await claude_integration._find_resumable_session(
                    update.effective_user.id, target_path
                )
                if existing:
                    session_id = existing.session_id
            context.user_data["claude_session_id"] = session_id

            is_git = (target_path / ".git").is_dir()
            git_badge = " (git)" if is_git else ""
            session_badge = " · session resumed" if session_id else ""

            await update.message.reply_text(
                f"Switched to <code>{escape_html(target_name)}/</code>"
                f"{git_badge}{session_badge}",
                parse_mode="HTML",
            )
            return

        # No args — list repos
        try:
            entries = sorted(
                [
                    d
                    for d in base.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ],
                key=lambda d: d.name,
            )
        except OSError as e:
            await update.message.reply_text(f"Error reading workspace: {e}")
            return

        if not entries:
            await update.message.reply_text(
                f"No repos in <code>{escape_html(str(base))}</code>.\n"
                'Clone one by telling me, e.g. <i>"clone org/repo"</i>.',
                parse_mode="HTML",
            )
            return

        lines: List[str] = []
        keyboard_rows: List[list] = []  # type: ignore[type-arg]
        current_name = current_dir.name if current_dir != base else None

        for d in entries:
            is_git = (d / ".git").is_dir()
            icon = "\U0001f4e6" if is_git else "\U0001f4c1"
            marker = " \u25c0" if d.name == current_name else ""
            lines.append(f"{icon} <code>{escape_html(d.name)}/</code>{marker}")

        # Build inline keyboard (2 per row)
        for i in range(0, len(entries), 2):
            row = []
            for j in range(2):
                if i + j < len(entries):
                    name = entries[i + j].name
                    row.append(InlineKeyboardButton(name, callback_data=f"cd:{name}"))
            keyboard_rows.append(row)

        reply_markup = InlineKeyboardMarkup(keyboard_rows)

        await update.message.reply_text(
            "<b>Repos</b>\n\n" + "\n".join(lines),
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def _handle_stop_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle stop: callbacks — interrupt a running Claude request."""
        query = update.callback_query
        target_user_id = int(query.data.split(":", 1)[1])

        # Only the requesting user can stop their own request
        if query.from_user.id != target_user_id:
            await query.answer(
                "Only the requesting user can stop this.", show_alert=True
            )
            return

        active = self._active_requests.get(target_user_id)
        if not active:
            await query.answer("Already completed.", show_alert=False)
            return
        if active.interrupted:
            await query.answer("Already stopping...", show_alert=False)
            return

        active.interrupt_event.set()
        active.interrupted = True
        await query.answer("Stopping...", show_alert=False)

        try:
            await active.progress_msg.edit_text("Stopping...", reply_markup=None)
        except Exception:
            pass

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks — switch directory and resume session if available."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, project_name = data.split(":", 1)

        base = self.settings.approved_directory
        new_path = base / project_name

        if not new_path.is_dir():
            await query.edit_message_text(
                f"Directory not found: <code>{escape_html(project_name)}</code>",
                parse_mode="HTML",
            )
            return

        context.user_data["current_directory"] = new_path

        # Look for a resumable session instead of always clearing
        claude_integration = context.bot_data.get("claude_integration")
        session_id = None
        if claude_integration:
            existing = await claude_integration._find_resumable_session(
                query.from_user.id, new_path
            )
            if existing:
                session_id = existing.session_id
        context.user_data["claude_session_id"] = session_id

        is_git = (new_path / ".git").is_dir()
        git_badge = " (git)" if is_git else ""
        session_badge = " · session resumed" if session_id else ""

        await query.edit_message_text(
            f"Switched to <code>{escape_html(project_name)}/</code>"
            f"{git_badge}{session_badge}",
            parse_mode="HTML",
        )

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=query.from_user.id,
                command="cd",
                args=[project_name],
                success=True,
            )

    async def agentic_projects(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Alias for /repo — list projects with inline switch buttons."""
        await self.agentic_repo(update, context)

    async def agentic_cost(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show today's spend, session totals, and daily limit (if set)."""
        user_id = update.effective_user.id
        lines: List[str] = ["<b>💰 Расход</b>", ""]

        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                status = rate_limiter.get_user_status(user_id)
                cost_usage = status.get("cost_usage", {}) or {}
                current = cost_usage.get("current", 0.0)
                limit = cost_usage.get("limit")
                remaining = cost_usage.get("remaining")
                lines.append(f"Сессия: <b>${current:.4f}</b>")
                if limit is not None:
                    lines.append(
                        f"Дневной лимит: <b>${limit:.2f}</b> "
                        f"(осталось ${remaining:.2f})"
                        if remaining is not None
                        else f"Дневной лимит: <b>${limit:.2f}</b>"
                    )
                req_usage = status.get("request_usage", {}) or {}
                if req_usage:
                    lines.append(
                        f"Запросов сегодня: <b>{req_usage.get('current', 0)}</b>"
                        f" / {req_usage.get('limit', '?')}"
                    )
            except Exception as exc:
                lines.append(f"<i>Rate limiter недоступен: {escape_html(str(exc))}</i>")
        else:
            lines.append("<i>Rate limiter не сконфигурирован.</i>")

        storage = context.bot_data.get("storage")
        if storage and getattr(storage, "costs", None):
            try:
                rows = await storage.costs.get_user_daily_costs(user_id, days=1)
                if rows:
                    today = rows[0]
                    lines.append("")
                    lines.append(
                        f"Всего за сегодня (БД): <b>${today.daily_cost:.4f}</b>"
                        f" · запросов: {today.request_count}"
                    )
            except Exception:
                pass

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="cost",
                args=[],
                success=True,
            )

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _agentic_menu_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle menu: callbacks from /start inline keyboard.

        Sets context.user_data["_invoked_from_menu"]=True so handlers that
        parse args from update.message.text know to ignore them (the fake
        message is the bot's own /start text, not a real user command).
        """
        query = update.callback_query
        await query.answer()
        action = query.data.split(":", 1)[1]

        # Build a fake Update pointing at the callback's message + same user
        fake_update = Update(update_id=update.update_id, message=query.message)
        # effective_user falls back to message.from_user; override to actual clicker
        fake_update._effective_user = query.from_user  # type: ignore[attr-defined]

        context.user_data["_invoked_from_menu"] = True
        try:
            if action == "projects":
                await self.agentic_repo(fake_update, context)
            elif action == "status":
                await self.agentic_status(fake_update, context)
            elif action == "cost":
                await self.agentic_cost(fake_update, context)
            elif action == "env":
                await self.agentic_env(fake_update, context)
            elif action == "settings":
                await self.agentic_settings(fake_update, context)
            elif action == "new":
                await self.agentic_new(fake_update, context)
            elif action == "model":
                await self.agentic_model(fake_update, context)
            elif action == "thinking":
                await self.agentic_thinking(fake_update, context)
            elif action == "timeout":
                await self.agentic_timeout(fake_update, context)
            elif action == "turns":
                await self.agentic_turns(fake_update, context)
            else:
                await query.message.reply_text(
                    f"Unknown menu action: <code>{escape_html(action)}</code>",
                    parse_mode="HTML",
                )
        finally:
            context.user_data.pop("_invoked_from_menu", None)

    # --- /env wizard ---

    async def agentic_env(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show current user-managed env vars and the 'add' button."""
        user_id = update.effective_user.id
        if self.settings.read_only_mode:
            await update.message.reply_text(
                "🔒 <b>Read-only mode</b>\n\n"
                "Этот бот работает в режиме только-чтение. "
                "Изменение переменных окружения недоступно.",
                parse_mode="HTML",
            )
            return
        rows = list_user_keys()
        lines: List[str] = ["<b>🔑 Переменные окружения</b>", ""]
        if rows:
            for key, masked in rows:
                lines.append(f"• <code>{escape_html(key)}</code> = {escape_html(masked)}")
        else:
            lines.append("<i>Пока ничего не добавлено.</i>")
        lines.append("")
        lines.append(
            "Защищённые переменные (бот не даёт менять): "
            + ", ".join(sorted(PROTECTED_KEYS))
        )
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("➕ Добавить", callback_data="env:add")],
            ]
        )
        await update.message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=keyboard,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id, command="env", args=[], success=True
            )

    async def _agentic_env_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle env:add, env:cancel callbacks."""
        query = update.callback_query
        await query.answer()
        action = query.data.split(":", 1)[1]

        if action == "add":
            if self.settings.read_only_mode:
                await query.message.reply_text(
                    "🔒 Read-only mode — добавление переменных недоступно."
                )
                return
            context.user_data["env_wizard_step"] = "waiting_name"
            context.user_data.pop("env_wizard_name", None)
            await query.message.reply_text(
                "Введи <b>имя</b> переменной (UPPER_CASE, латиница+цифры+_).\n"
                "Например: <code>FIGJAM_TOKEN</code>, "
                "<code>DEEPGRAM_API_KEY</code>, <code>GITHUB_TOKEN_NASTYA</code>.\n\n"
                "Защищённые (нельзя): "
                + ", ".join(sorted(PROTECTED_KEYS))
                + "\n\nОтмена: /env_cancel",
                parse_mode="HTML",
            )
        elif action == "cancel":
            context.user_data.pop("env_wizard_step", None)
            context.user_data.pop("env_wizard_name", None)
            await query.message.reply_text("Отменено.")
        else:
            await query.message.reply_text(
                f"Unknown env action: <code>{escape_html(action)}</code>",
                parse_mode="HTML",
            )

    # --- /model and /thinking ---

    _CLAUDE_MODELS = [
        ("claude-opus-4-8", "Opus 4.8 — новейшая"),
        ("claude-opus-4-7", "Opus 4.7"),
        ("claude-sonnet-4-6", "Sonnet 4.6"),
        ("claude-haiku-4-5", "Haiku 4.5"),
    ]

    _THINKING_LEVELS = [
        ("low", "Low — минимум"),
        ("medium", "Medium"),
        ("high", "High"),
        ("xhigh", "Extra High — максимум"),
    ]

    def _apply_runtime_setting(self, attr: str, value: str, env_key: str) -> None:
        """In-memory patch + persist to .env so it survives restart."""
        setattr(self.settings, attr, value)
        try:
            write_env_var(env_key, value)
        except Exception as e:
            logger.error("Failed to persist env var", key=env_key, error=str(e))

    async def agentic_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/model — show current model + picker, or set via arg."""
        user_id = update.effective_user.id
        if context.user_data.get("_invoked_from_menu"):
            args = []
        else:
            args = update.message.text.split()[1:] if update.message.text else []
        valid_ids = {mid for mid, _ in self._CLAUDE_MODELS}
        if args:
            model_id = args[0].strip()
            if model_id not in valid_ids:
                await update.message.reply_text(
                    "Неизвестная модель. Доступные: " + ", ".join(sorted(valid_ids)),
                )
                return
            self._apply_runtime_setting("claude_model", model_id, "CLAUDE_MODEL")
            await update.message.reply_text(
                "Модель → <code>" + escape_html(model_id) + "</code>",
                parse_mode="HTML",
            )
            return
        current = self.settings.claude_model or "не задана"
        rows = []
        for mid, label in self._CLAUDE_MODELS:
            marker = " ✓" if mid == self.settings.claude_model else ""
            rows.append([InlineKeyboardButton(label + marker, callback_data="model:" + mid)])
        keyboard = InlineKeyboardMarkup(rows)
        text = ("<b>🤖 Модель</b>" + chr(10) + chr(10) +
                "Текущая: <code>" + escape_html(str(current)) + "</code>" + chr(10) + chr(10) +
                "Выбери:")
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(user_id=user_id, command="model", args=args, success=True)

    async def _agentic_model_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle model:<id> callback buttons."""
        query = update.callback_query
        await query.answer()
        model_id = query.data.split(":", 1)[1]
        valid_ids = {mid for mid, _ in self._CLAUDE_MODELS}
        if model_id not in valid_ids:
            await query.message.reply_text("Неизвестная модель.")
            return
        self._apply_runtime_setting("claude_model", model_id, "CLAUDE_MODEL")
        await query.edit_message_text(
            "Модель → <code>" + escape_html(model_id) + "</code>",
            parse_mode="HTML",
        )

    async def agentic_thinking(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/thinking — show current effort + picker, or set via arg."""
        user_id = update.effective_user.id
        if context.user_data.get("_invoked_from_menu"):
            args = []
        else:
            args = update.message.text.split()[1:] if update.message.text else []
        valid_levels = {lvl for lvl, _ in self._THINKING_LEVELS}
        if args:
            level = args[0].strip().lower()
            if level not in valid_levels:
                await update.message.reply_text("Уровни: low, medium, high, xhigh")
                return
            self._apply_runtime_setting("claude_thinking_effort", level, "CLAUDE_THINKING_EFFORT")
            await update.message.reply_text(
                "Thinking → <code>" + escape_html(level) + "</code>",
                parse_mode="HTML",
            )
            return
        current = self.settings.claude_thinking_effort
        rows = []
        for lvl, label in self._THINKING_LEVELS:
            marker = " ✓" if lvl == current else ""
            rows.append([InlineKeyboardButton(label + marker, callback_data="thinking:" + lvl)])
        keyboard = InlineKeyboardMarkup(rows)
        text = ("<b>🧠 Thinking effort</b>" + chr(10) + chr(10) +
                "Текущий: <code>" + escape_html(current) + "</code>" + chr(10) + chr(10) +
                "Выбери:")
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(user_id=user_id, command="thinking", args=args, success=True)

    async def _agentic_thinking_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle thinking:<level> callback buttons."""
        query = update.callback_query
        await query.answer()
        level = query.data.split(":", 1)[1]
        valid_levels = {lvl for lvl, _ in self._THINKING_LEVELS}
        if level not in valid_levels:
            await query.message.reply_text("Неизвестный уровень.")
            return
        self._apply_runtime_setting("claude_thinking_effort", level, "CLAUDE_THINKING_EFFORT")
        await query.edit_message_text(
            "Thinking → <code>" + escape_html(level) + "</code>",
            parse_mode="HTML",
        )

    # --- /timeout and /turns ---

    _TIMEOUT_OPTIONS = [
        (300, "5 мин"),
        (900, "15 мин"),
        (1800, "30 мин"),
        (3600, "1 час"),
        (7200, "2 часа"),
    ]

    _TURNS_OPTIONS = [
        (50, "50"),
        (100, "100"),
        (200, "200"),
        (500, "500"),
    ]

    async def agentic_timeout(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/timeout — picker for Claude operation timeout (seconds)."""
        user_id = update.effective_user.id
        if context.user_data.get("_invoked_from_menu"):
            args = []
        else:
            args = update.message.text.split()[1:] if update.message.text else []
        if args:
            try:
                seconds = int(args[0])
            except ValueError:
                await update.message.reply_text("Передай число секунд, например /timeout 3600")
                return
            if seconds < 60 or seconds > 14400:
                await update.message.reply_text("Диапазон: 60 — 14400 секунд (от минуты до 4 часов).")
                return
            setattr(self.settings, "claude_timeout_seconds", seconds)
            try:
                write_env_var("CLAUDE_TIMEOUT_SECONDS", str(seconds))
            except Exception as e:
                logger.error("Failed to persist CLAUDE_TIMEOUT_SECONDS", error=str(e))
            await update.message.reply_text(
                "Timeout → <b>" + str(seconds) + "</b> сек",
                parse_mode="HTML",
            )
            return
        current = self.settings.claude_timeout_seconds
        rows = []
        for v, label in self._TIMEOUT_OPTIONS:
            marker = " ✓" if v == current else ""
            rows.append([InlineKeyboardButton(label + marker, callback_data="timeout:" + str(v))])
        keyboard = InlineKeyboardMarkup(rows)
        text = ("<b>⏱ Timeout</b>" + chr(10) + chr(10) +
                "Текущий: <b>" + str(current) + "</b> сек" + chr(10) + chr(10) +
                "Выбери:")
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(user_id=user_id, command="timeout", args=args, success=True)

    async def _agentic_timeout_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle timeout:<seconds> callback buttons."""
        query = update.callback_query
        await query.answer()
        try:
            seconds = int(query.data.split(":", 1)[1])
        except ValueError:
            await query.message.reply_text("Bad value")
            return
        setattr(self.settings, "claude_timeout_seconds", seconds)
        try:
            write_env_var("CLAUDE_TIMEOUT_SECONDS", str(seconds))
        except Exception as e:
            logger.error("Failed to persist CLAUDE_TIMEOUT_SECONDS", error=str(e))
        await query.edit_message_text(
            "Timeout → <b>" + str(seconds) + "</b> сек",
            parse_mode="HTML",
        )

    async def agentic_turns(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """/turns — picker for Claude max conversation turns."""
        user_id = update.effective_user.id
        if context.user_data.get("_invoked_from_menu"):
            args = []
        else:
            args = update.message.text.split()[1:] if update.message.text else []
        if args:
            try:
                turns = int(args[0])
            except ValueError:
                await update.message.reply_text("Передай число, например /turns 200")
                return
            if turns < 1 or turns > 2000:
                await update.message.reply_text("Диапазон: 1 — 2000.")
                return
            setattr(self.settings, "claude_max_turns", turns)
            try:
                write_env_var("CLAUDE_MAX_TURNS", str(turns))
            except Exception as e:
                logger.error("Failed to persist CLAUDE_MAX_TURNS", error=str(e))
            await update.message.reply_text(
                "Max turns → <b>" + str(turns) + "</b>",
                parse_mode="HTML",
            )
            return
        current = self.settings.claude_max_turns
        rows = []
        for v, label in self._TURNS_OPTIONS:
            marker = " ✓" if v == current else ""
            rows.append([InlineKeyboardButton(label + marker, callback_data="turns:" + str(v))])
        keyboard = InlineKeyboardMarkup(rows)
        text = ("<b>🔁 Max turns</b>" + chr(10) + chr(10) +
                "Текущее: <b>" + str(current) + "</b>" + chr(10) + chr(10) +
                "Выбери:")
        await update.message.reply_text(text, parse_mode="HTML", reply_markup=keyboard)
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(user_id=user_id, command="turns", args=args, success=True)

    async def _agentic_turns_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle turns:<n> callback buttons."""
        query = update.callback_query
        await query.answer()
        try:
            turns = int(query.data.split(":", 1)[1])
        except ValueError:
            await query.message.reply_text("Bad value")
            return
        setattr(self.settings, "claude_max_turns", turns)
        try:
            write_env_var("CLAUDE_MAX_TURNS", str(turns))
        except Exception as e:
            logger.error("Failed to persist CLAUDE_MAX_TURNS", error=str(e))
        await query.edit_message_text(
            "Max turns → <b>" + str(turns) + "</b>",
            parse_mode="HTML",
        )

    async def _handle_env_wizard_input(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        text: str,
    ) -> None:
        """Process the user's typed input during /env wizard."""
        step = context.user_data.get("env_wizard_step")
        text = (text or "").strip()

        # Allow /env_cancel-like escape via plain text
        if text.lower() in {"отмена", "cancel", "/cancel", "/env_cancel"}:
            context.user_data.pop("env_wizard_step", None)
            context.user_data.pop("env_wizard_name", None)
            await update.message.reply_text("Отменено.")
            return

        if step == "waiting_name":
            if not is_valid_key(text):
                await update.message.reply_text(
                    "❌ Невалидное имя. Только UPPER_CASE, латиница, цифры, "
                    "подчёркивания. И нельзя защищённые. Попробуй ещё раз "
                    "или напиши «отмена»."
                )
                return
            context.user_data["env_wizard_name"] = text
            context.user_data["env_wizard_step"] = "waiting_value"
            await update.message.reply_text(
                f"Имя: <code>{escape_html(text)}</code>\n\n"
                f"Теперь пришли <b>значение</b> (токен/ключ).\n"
                f"⚠️ Твоё сообщение со значением будет <b>сразу удалено</b> "
                f"из чата для безопасности. Значение запишется в .env.",
                parse_mode="HTML",
            )
            return

        if step == "waiting_value":
            key = context.user_data.get("env_wizard_name", "")
            # Delete the user's message ASAP (best-effort)
            try:
                await update.message.delete()
            except Exception as exc:
                logger.warning(
                    "env_wizard: failed to delete value message",
                    error=str(exc),
                    user_id=update.effective_user.id,
                )

            if not key:
                context.user_data.pop("env_wizard_step", None)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Что-то пошло не так — имя потерялось. Начни заново через /env.",
                )
                return

            if len(text) > MAX_VALUE_LEN:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"❌ Слишком длинное значение ({len(text)} > {MAX_VALUE_LEN}). "
                    f"Пришли покороче или /env_cancel.",
                )
                return

            try:
                write_env_var(key, text)
            except Exception as exc:
                logger.error(
                    "env_wizard: write failed",
                    error=str(exc),
                    key=key,
                )
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=f"❌ Не удалось записать в .env: <code>{escape_html(str(exc))}</code>",
                    parse_mode="HTML",
                )
                context.user_data.pop("env_wizard_step", None)
                context.user_data.pop("env_wizard_name", None)
                return

            context.user_data.pop("env_wizard_step", None)
            context.user_data.pop("env_wizard_name", None)

            audit_logger = context.bot_data.get("audit_logger")
            if audit_logger:
                await audit_logger.log_command(
                    user_id=update.effective_user.id,
                    command="env_add",
                    args=[key],
                    success=True,
                )

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    f"✅ <code>{escape_html(key)}</code> сохранено в .env "
                    f"(твоё сообщение удалено).\n\n"
                    f"Для применения нужен restart: пришли <code>/restart</code> "
                    f"или попроси меня выполнить <code>sudo systemctl restart agent-vps</code>."
                ),
                parse_mode="HTML",
            )
            return

        # Unknown state — clear it just in case
        context.user_data.pop("env_wizard_step", None)
        context.user_data.pop("env_wizard_name", None)

    # --- /settings overview ---

    async def agentic_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Settings panel: shows current config + navigation buttons."""
        user_id = update.effective_user.id
        settings = self.settings

        current_dir = context.user_data.get(
            "current_directory", settings.approved_directory
        )
        verbose_level = self._get_verbose_level(context)
        verbose_label = {0: "quiet", 1: "normal", 2: "detailed"}.get(
            verbose_level, "?"
        )

        # User-managed env vars count
        env_count = len(list_user_keys())

        # Subscription / auth indicator: ANTHROPIC_API_KEY set?
        api_key = getattr(settings, "anthropic_api_key", None)
        if api_key:
            auth_line = "API key (billed by Anthropic)"
        else:
            auth_line = "Claude Code OAuth (Max/Pro subscription)"

        voice_provider = getattr(settings, "voice_provider", "—")
        voice_enabled = getattr(settings, "enable_voice_messages", False)
        voice_line = (
            f"{voice_provider}" if voice_enabled else "off"
        )

        ro_line = "🔒 <b>Read-only mode</b>" if settings.read_only_mode else "🔓 Full access"
        lines: List[str] = [
            "<b>⚙️ Настройки</b>",
            "",
            f"📂 Working dir: <code>{escape_html(str(current_dir))}</code>",
            f"🔌 Подключение: {auth_line}",
            f"{ro_line}",
            f"🤖 Agentic mode: {settings.agentic_mode}",
            f"🧬 Model: <code>{escape_html(str(settings.claude_model or '—'))}</code>",
            f"🧠 Thinking: <b>{settings.claude_thinking_effort}</b>",
            f"⏱ Timeout: <b>{settings.claude_timeout_seconds}</b>s",
            f"🔁 Max turns: <b>{settings.claude_max_turns}</b>",
            f"📢 Verbose: <b>{verbose_level}</b> ({verbose_label})",
            f"🎤 Voice: {voice_line}",
            f"🔑 User env vars: <b>{env_count}</b>",
            f"🧵 Project threads: {settings.enable_project_threads}",
        ]

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("🔑 Ключи", callback_data="menu:env"),
                    InlineKeyboardButton("💰 Расход", callback_data="menu:cost"),
                ],
                [
                    InlineKeyboardButton("📊 Статус", callback_data="menu:status"),
                    InlineKeyboardButton("📁 Проекты", callback_data="menu:projects"),
                ],
                [
                    InlineKeyboardButton("🧬 Модель", callback_data="menu:model"),
                    InlineKeyboardButton("🧠 Thinking", callback_data="menu:thinking"),
                ],
                [
                    InlineKeyboardButton("⏱ Timeout", callback_data="menu:timeout"),
                    InlineKeyboardButton("🔁 Turns", callback_data="menu:turns"),
                ],
            ]
        )

        await update.message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=keyboard,
        )

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id, command="settings", args=[], success=True
            )

    def _maybe_prefix_readonly(self, prompt: str) -> str:
        """In read_only_mode, inject a guardrail prefix into every user prompt.

        Not a hard security boundary (Claude can still call tools), but it
        primes the assistant to refuse destructive actions. Hard enforcement
        belongs in ToolMonitor; this is the cheap signal.
        """
        if not self.settings.read_only_mode:
            return prompt
        guard = (
            "[SYSTEM CONSTRAINT: READ_ONLY_MODE — "
            "this bot is shared with a partner who only has read access. "
            "DO NOT use Write/Edit/MultiEdit/NotebookEdit tools. "
            "DO NOT run shell commands that modify files (rm, mv, dd, chmod, "
            "chown, truncate, tee, cp into protected paths). "
            "DO NOT modify .env, .git, systemd units, or anything under /etc /opt /usr. "
            "Read-only inspection is fine: ls, cat, grep, find, git log/diff/status. "
            "If the user requests a destructive action, refuse politely and explain "
            "that they need to ask the bot's owner.]\n\n"
        )
        return guard + prompt

    async def agentic_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Global stop command: interrupt the user's currently running Claude task.

        Equivalent to tapping the inline "Stop" button under the latest
        Working... message, but available as a regular command from the menu
        or by typing /stop.
        """
        user_id = update.effective_user.id
        active = self._active_requests.get(user_id)
        if not active:
            await update.message.reply_text(
                "Нет активной задачи. Тебе нечего останавливать."
            )
            return
        if active.interrupted:
            await update.message.reply_text("Уже останавливается...")
            return

        active.interrupt_event.set()
        active.interrupted = True
        try:
            await active.progress_msg.edit_text("Stopping...", reply_markup=None)
        except Exception:
            pass
        await update.message.reply_text("🛑 Сигнал остановки отправлен.")

        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id, command="stop", args=[], success=True
            )
