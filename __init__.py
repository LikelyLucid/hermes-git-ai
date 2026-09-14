"""Git AI attribution hooks for Hermes Agent.

The plugin intentionally has no dependencies outside the Python standard
library. It is a thin adapter from Hermes' plugin hook payloads to Git AI's
versioned ``agent-v1`` checkpoint input.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


_LOG = logging.getLogger(__name__)
_LOCK = threading.RLock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_REPO_CACHE: dict[str, str] = {}
_MISSING_GIT_AI_LOGGED = False

_FILE_TOOLS = frozenset(
    {
        "write_file",
        "patch",
        "apply_patch",
        "edit_file",
        "create_file",
        "delete_file",
        "move_file",
        "rename_file",
    }
)
_NON_LOCAL_ENVIRONMENTS = frozenset(
    {
        "docker",
        "modal",
        "managed_modal",
        "vercel_sandbox",
        "singularity",
        "daytona",
        "ssh",
        "remote",
    }
)
_V4A_FILE_HEADER = re.compile(
    r"^\*\*\*\s*(?:Update|Add|Delete|Move)\s+File:\s*(.+?)\s*$",
    re.MULTILINE,
)
_V4A_MOVE_HEADER = re.compile(r"^\*\*\*\s*Move\s+to:\s*(.+?)\s*$", re.MULTILINE)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _transcript_enabled() -> bool:
    return str(os.environ.get("HERMES_GIT_AI_TRANSCRIPT", "on")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _integration_disabled() -> bool:
    return _truthy(os.environ.get("HERMES_GIT_AI_DISABLED"))


def _is_local_environment() -> bool:
    environment = str(os.environ.get("TERMINAL_ENV", "local")).strip().lower()
    return environment not in _NON_LOCAL_ENVIRONMENTS


def _checkpoint_timeout() -> float:
    raw = os.environ.get("HERMES_GIT_AI_TIMEOUT", "10")
    try:
        return max(1.0, min(float(raw), 120.0))
    except (TypeError, ValueError):
        return 10.0


def _git_ai_command() -> list[str] | None:
    """Return the executable prefix, or None when Git AI is unavailable."""
    if _integration_disabled():
        return None

    override = str(os.environ.get("HERMES_GIT_AI_BIN", "")).strip()
    if override:
        resolved = shutil.which(override) or (override if Path(override).is_file() else None)
        if resolved:
            return [resolved]
        return None

    executable = shutil.which("git-ai")
    if executable:
        # Prefer the documented Git extension spelling. The direct executable
        # fallback keeps the bridge usable when Git itself is not on PATH.
        git_executable = shutil.which("git")
        return [git_executable, "ai"] if git_executable else [executable]

    global _MISSING_GIT_AI_LOGGED
    if not _MISSING_GIT_AI_LOGGED:
        _LOG.info("Git AI integration is dormant: git-ai is not on PATH")
        _MISSING_GIT_AI_LOGGED = True
    return None


def _run_checkpoint(payload: Mapping[str, Any], cwd: str | Path) -> bool:
    """Send one checkpoint to Git AI without allowing it to break Hermes."""
    command = _git_ai_command()
    if command is None or _integration_disabled() or not _is_local_environment():
        return False

    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        _LOG.warning("Git AI payload could not be encoded: %s", exc)
        return False

    invocation = command + ["checkpoint", "agent-v1", "--hook-input", "stdin"]
    try:
        completed = subprocess.run(
            invocation,
            cwd=str(cwd),
            input=encoded,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_checkpoint_timeout(),
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _LOG.warning("Git AI checkpoint failed open: %s", type(exc).__name__)
        return False

    if completed.returncode != 0:
        _LOG.debug("Git AI checkpoint exited with status %s", completed.returncode)
        return False
    return True


def _absolute_path(raw: Any, task_id: str = "") -> Path | None:
    if not isinstance(raw, (str, os.PathLike)):
        return None
    value = os.fspath(raw).strip()
    if not value:
        return None

    # Hermes' file layer knows the task-specific workspace (including worktree
    # overrides). Reuse that resolver when available, but keep this plugin
    # usable during discovery and in standalone tests.
    try:
        from tools.file_tools import _resolve_path  # type: ignore

        resolved = _resolve_path(value, task_id or "default")
        value = os.fspath(resolved)
    except Exception:
        expanded = os.path.expandvars(os.path.expanduser(value))
        if not os.path.isabs(expanded):
            base = (
                os.environ.get("TERMINAL_CWD")
                or os.environ.get("HERMES_TERMINAL_CWD")
                or os.getcwd()
            )
            value = os.path.join(base, expanded)
        else:
            value = expanded

    # abspath normalizes ``..`` without dereferencing symlinks. That preserves
    # the path namespace used by local worktrees and by the file tool.
    return Path(os.path.abspath(os.fspath(value)))


def _terminal_cwd(args: Mapping[str, Any], session_id: str, task_id: str) -> Path:
    raw = args.get("workdir") or args.get("cwd")
    if isinstance(raw, (str, os.PathLike)) and os.fspath(raw).strip():
        resolved = _absolute_path(raw, task_id)
        if resolved is not None:
            return resolved

    # terminal_tool records a per-session cwd when the model does not provide a
    # workdir on every call. This is the same state used by the real terminal.
    try:
        from tools.terminal_tool import get_session_cwd  # type: ignore

        recorded = get_session_cwd(session_id or None)
        if recorded:
            resolved = _absolute_path(recorded, task_id)
            if resolved is not None:
                return resolved
    except Exception:
        pass

    resolved = _absolute_path(
        os.environ.get("TERMINAL_CWD")
        or os.environ.get("HERMES_TERMINAL_CWD")
        or os.getcwd(),
        task_id,
    )
    return resolved or Path.cwd()


def _repo_root(path: Path) -> str | None:
    candidate = path if path.is_dir() else path.parent
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    cache_key = os.fspath(candidate)
    with _LOCK:
        cached = _REPO_CACHE.get(cache_key)
    if cached:
        return cached

    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(candidate), "rev-parse", "--show-toplevel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    root = completed.stdout.strip()
    if not root:
        return None
    root = os.path.abspath(os.path.expanduser(root))
    with _LOCK:
        _REPO_CACHE[cache_key] = root
    return root


def _patch_paths(args: Mapping[str, Any], task_id: str) -> list[Path]:
    raw_paths: list[Any] = []
    for key in ("path", "filepath", "file_path"):
        if args.get(key):
            raw_paths.append(args[key])

    patch_text = args.get("patch")
    mode = str(args.get("mode", "replace")).strip().lower()
    if isinstance(patch_text, str) and (mode == "patch" or not raw_paths):
        raw_paths.extend(match.group(1) for match in _V4A_FILE_HEADER.finditer(patch_text))
        raw_paths.extend(match.group(1) for match in _V4A_MOVE_HEADER.finditer(patch_text))

    # Some external file tools expose a list of files rather than one path.
    for key in ("paths", "filepaths", "file_paths", "files"):
        value = args.get(key)
        if isinstance(value, (list, tuple)):
            raw_paths.extend(value)

    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_paths:
        resolved = _absolute_path(raw, task_id)
        if resolved is None:
            continue
        value = os.fspath(resolved)
        if value not in seen:
            seen.add(value)
            paths.append(resolved)
    return paths[:1000]


def _group_file_paths(paths: Sequence[Path]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for path in paths:
        root = _repo_root(path)
        if root is None:
            continue
        grouped.setdefault(root, []).append(os.fspath(path))
    return grouped


def _timestamp(message: Mapping[str, Any]) -> str:
    for key in ("timestamp", "created_at", "time"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_text(item) for item in value]
        return "".join(part for part in parts if part)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        if "content" in value:
            return _text(value["content"])
    return ""


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except (TypeError, ValueError):
                return value
        return value
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _tool_name(call: Mapping[str, Any]) -> str:
    function = call.get("function")
    if isinstance(function, Mapping) and function.get("name"):
        return str(function["name"])
    return str(call.get("name") or call.get("tool_name") or "unknown_tool")


def _tool_input(call: Mapping[str, Any]) -> Any:
    function = call.get("function")
    if isinstance(function, Mapping) and "arguments" in function:
        return _json_value(function["arguments"])
    for key in ("input", "arguments", "args"):
        if key in call:
            return _json_value(call[key])
    return {}


def _transcript_message(message: Any) -> list[dict[str, Any]]:
    if not isinstance(message, Mapping):
        return []
    role = str(message.get("role") or "").strip().lower()
    stamp = _timestamp(message)
    if role == "user":
        text = _text(message.get("content"))
        return [{"type": "user", "text": text, "timestamp": stamp}] if text else []
    if role == "assistant":
        result: list[dict[str, Any]] = []
        text = _text(message.get("content"))
        if text:
            result.append({"type": "assistant", "text": text, "timestamp": stamp})
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, Mapping):
                    result.append(
                        {
                            "type": "tool_use",
                            "name": _tool_name(call),
                            "input": _tool_input(call),
                            "timestamp": stamp,
                        }
                    )
        return result
    # Tool results and system messages are intentionally omitted. Git AI's
    # integration guide recommends excluding tool results because they are
    # large, stale, and may contain unrelated command output.
    return []


def _normalize_transcript(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, (list, tuple)):
        return []
    result: list[dict[str, Any]] = []
    for message in history:
        result.extend(_transcript_message(message))
    return result


def _session_key(session_id: Any, task_id: Any) -> str:
    sid = str(session_id or "").strip()
    if sid:
        return sid
    tid = str(task_id or "").strip()
    if tid:
        return "task:" + tid
    return "thread:" + str(threading.get_ident())


def _get_state(session_id: Any, task_id: Any) -> dict[str, Any]:
    key = _session_key(session_id, task_id)
    with _LOCK:
        state = _SESSIONS.get(key)
        if state is None:
            state = {
                "conversation_id": str(session_id or task_id or ("hermes-" + uuid.uuid4().hex)),
                "model": "unknown",
                "transcript": [],
            }
            _SESSIONS[key] = state
        return state


def _refresh_state(
    session_id: Any,
    task_id: Any,
    *,
    history: Any = None,
    user_message: Any = None,
    assistant_response: Any = None,
    model: Any = None,
) -> dict[str, Any]:
    state = _get_state(session_id, task_id)
    with _LOCK:
        if isinstance(model, str) and model.strip():
            state["model"] = model.strip()
        normalized = _normalize_transcript(history)
        if normalized:
            state["transcript"] = normalized
        current_user = _text(user_message)
        if current_user:
            users = [item for item in state["transcript"] if item.get("type") == "user"]
            if not users or users[-1].get("text") != current_user:
                state["transcript"].append(
                    {
                        "type": "user",
                        "text": current_user,
                        "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                )
        response_text = _text(assistant_response)
        if response_text:
            if not state["transcript"] or not (
                state["transcript"][-1].get("type") == "assistant"
                and state["transcript"][-1].get("text") == response_text
            ):
                state["transcript"].append(
                    {
                        "type": "assistant",
                        "text": response_text,
                        "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                        .isoformat()
                        .replace("+00:00", "Z"),
                    }
                )
        return state


def _snapshot_state(session_id: Any, task_id: Any) -> tuple[str, str, str, list[dict[str, Any]]]:
    state = _get_state(session_id, task_id)
    with _LOCK:
        return (
            str(state["conversation_id"]),
            str(state.get("model") or "unknown"),
            str(os.environ.get("HERMES_GIT_AI_AGENT_NAME", "hermes")).strip() or "hermes",
            copy.deepcopy(state.get("transcript") or []),
        )


def _record_tool_use(session_id: Any, task_id: Any, tool_name: Any, args: Any) -> None:
    name = str(tool_name or "unknown_tool")
    safe_args = _json_value(args if isinstance(args, Mapping) else {})
    state = _get_state(session_id, task_id)
    with _LOCK:
        state["transcript"].append(
            {
                "type": "tool_use",
                "name": name,
                "input": safe_args,
                "timestamp": _datetime.datetime.now(_datetime.timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
            }
        )


def _payload_with_transcript(
    payload: dict[str, Any], transcript: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if _transcript_enabled() and transcript:
        payload["transcript"] = {"messages": list(transcript)}
    return payload


def _file_checkpoint(
    *,
    phase: str,
    tool_name: str,
    args: Mapping[str, Any],
    session_id: Any,
    task_id: Any,
) -> None:
    if not _is_local_environment():
        return
    paths = _patch_paths(args, str(task_id or ""))
    if not paths:
        return
    grouped = _group_file_paths(paths)
    if not grouped:
        return

    conversation_id, model, agent_name, transcript = _snapshot_state(session_id, task_id)
    for root, filepaths in grouped.items():
        if phase == "pre":
            payload: dict[str, Any] = {
                "type": "human",
                "repo_working_dir": root,
                "will_edit_filepaths": filepaths,
            }
        else:
            payload = {
                "type": "ai_agent",
                "repo_working_dir": root,
                "edited_filepaths": filepaths,
                "agent_name": agent_name,
                "model": model,
                "conversation_id": conversation_id,
                "tool_use_id": str(args.get("tool_call_id") or "") or None,
            }
            payload = _payload_with_transcript(payload, transcript)
        _run_checkpoint(payload, root)


def _shell_checkpoint(
    *,
    phase: str,
    args: Mapping[str, Any],
    session_id: Any,
    task_id: Any,
    tool_call_id: Any,
) -> None:
    if not _is_local_environment():
        return
    cwd = _terminal_cwd(args, str(session_id or ""), str(task_id or ""))
    if _repo_root(cwd) is None:
        return
    conversation_id, model, agent_name, transcript = _snapshot_state(session_id, task_id)
    payload: dict[str, Any] = {
        "type": "pre_shell_command" if phase == "pre" else "post_shell_command",
        "repo_working_dir": os.fspath(cwd),
        "agent_name": agent_name,
        "model": model,
        "conversation_id": conversation_id,
        "tool_use_id": str(tool_call_id or "") or None,
        "command": str(args.get("command") or ""),
    }
    payload = _payload_with_transcript(payload, transcript)
    _run_checkpoint(payload, cwd)


def _tool_succeeded(status: Any, result: Any) -> bool:
    normalized = str(status or "").strip().lower()
    if normalized and normalized not in {"ok", "success", "completed"}:
        return False
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            parsed = result
    return not (isinstance(parsed, Mapping) and parsed.get("error"))


def _on_pre_llm_call(**kwargs: Any) -> None:
    _refresh_state(
        kwargs.get("session_id"),
        kwargs.get("task_id"),
        history=kwargs.get("conversation_history"),
        user_message=kwargs.get("user_message"),
        model=kwargs.get("model"),
    )


def _on_post_llm_call(**kwargs: Any) -> None:
    _refresh_state(
        kwargs.get("session_id"),
        kwargs.get("task_id"),
        history=kwargs.get("conversation_history"),
        user_message=kwargs.get("user_message"),
        assistant_response=kwargs.get("assistant_response"),
        model=kwargs.get("model"),
    )


def _on_pre_tool_call(**kwargs: Any) -> None:
    tool_name = str(kwargs.get("tool_name") or "")
    args = kwargs.get("args")
    args = args if isinstance(args, Mapping) else {}
    session_id = kwargs.get("session_id")
    task_id = kwargs.get("task_id")
    _record_tool_use(session_id, task_id, tool_name, args)

    if tool_name == "terminal":
        _shell_checkpoint(
            phase="pre",
            args=args,
            session_id=session_id,
            task_id=task_id,
            tool_call_id=kwargs.get("tool_call_id"),
        )
    elif tool_name in _FILE_TOOLS:
        _file_checkpoint(
            phase="pre",
            tool_name=tool_name,
            args=args,
            session_id=session_id,
            task_id=task_id,
        )


def _on_post_tool_call(**kwargs: Any) -> None:
    tool_name = str(kwargs.get("tool_name") or "")
    args = kwargs.get("args")
    args = args if isinstance(args, Mapping) else {}
    if not _tool_succeeded(kwargs.get("status"), kwargs.get("result")):
        return

    session_id = kwargs.get("session_id")
    task_id = kwargs.get("task_id")
    if tool_name == "terminal":
        _shell_checkpoint(
            phase="post",
            args=args,
            session_id=session_id,
            task_id=task_id,
            tool_call_id=kwargs.get("tool_call_id"),
        )
    elif tool_name in _FILE_TOOLS:
        _file_checkpoint(
            phase="post",
            tool_name=tool_name,
            args=args,
            session_id=session_id,
            task_id=task_id,
        )


def _drop_session(**kwargs: Any) -> None:
    candidates = {
        str(kwargs.get(key) or "").strip()
        for key in ("session_id", "old_session_id", "new_session_id")
    }
    candidates.discard("")
    with _LOCK:
        for key in candidates:
            _SESSIONS.pop(key, None)


def register(ctx: Any) -> None:
    """Register the Git AI bridge with Hermes' native plugin manager."""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("on_session_end", _drop_session)
    ctx.register_hook("on_session_reset", _drop_session)
