from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "__init__.py"
_spec = importlib.util.spec_from_file_location("git_ai_hermes_plugin", PLUGIN_PATH)
assert _spec and _spec.loader
plugin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plugin)


class _Context:
    def __init__(self):
        self.hooks: dict[str, object] = {}

    def register_hook(self, name, callback):
        self.hooks[name] = callback


class GitAiPluginTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        (self.repo / "src").mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.calls: list[tuple[dict, str]] = []
        plugin._SESSIONS.clear()
        plugin._REPO_CACHE.clear()
        self.env = patch.dict(
            os.environ,
            {
                "TERMINAL_CWD": str(self.repo),
                "HERMES_GIT_AI_TRANSCRIPT": "on",
                "TERMINAL_ENV": "local",
            },
            clear=False,
        )
        self.env.start()
        self.runner = patch.object(plugin, "_run_checkpoint", side_effect=self._record)
        self.runner.start()

    def tearDown(self):
        self.runner.stop()
        self.env.stop()
        plugin._SESSIONS.clear()
        plugin._REPO_CACHE.clear()
        self.tempdir.cleanup()

    def _record(self, payload, cwd):
        self.calls.append((dict(payload), str(cwd)))
        return True

    def _seed_session(self):
        plugin._on_pre_llm_call(
            session_id="session-1",
            task_id="task-1",
            user_message="Add the new behavior",
            conversation_history=[
                {
                    "role": "user",
                    "content": "Add the new behavior",
                    "timestamp": "2026-09-14T00:00:00Z",
                }
            ],
            model="gpt-5.6-luna",
        )

    def test_registers_the_required_hooks(self):
        context = _Context()
        plugin.register(context)
        self.assertEqual(
            set(context.hooks),
            {
                "pre_llm_call",
                "post_llm_call",
                "pre_tool_call",
                "post_tool_call",
                "on_session_end",
                "on_session_reset",
            },
        )

    def test_file_edit_is_human_before_and_ai_after(self):
        self._seed_session()
        args = {"path": "src/main.py", "content": "print('hello')\n"}
        plugin._on_pre_tool_call(
            tool_name="write_file",
            args=args,
            session_id="session-1",
            task_id="task-1",
            tool_call_id="file-call-1",
        )
        plugin._on_post_tool_call(
            tool_name="write_file",
            args=args,
            result=json.dumps({"ok": True}),
            status="ok",
            session_id="session-1",
            task_id="task-1",
            tool_call_id="file-call-1",
        )

        self.assertEqual(len(self.calls), 2)
        before, after = (payload for payload, _ in self.calls)
        self.assertEqual(before["type"], "human")
        self.assertEqual(after["type"], "ai_agent")
        self.assertEqual(after["agent_name"], "hermes")
        self.assertEqual(after["model"], "gpt-5.6-luna")
        self.assertEqual(after["conversation_id"], "session-1")
        self.assertEqual(
            after["edited_filepaths"],
            [str(self.repo / "src/main.py")],
        )
        transcript_types = [item["type"] for item in after["transcript"]["messages"]]
        self.assertEqual(transcript_types, ["user", "tool_use"])

    def test_v4a_patch_paths_are_forwarded(self):
        self._seed_session()
        args = {
            "mode": "patch",
            "patch": (
                "*** Begin Patch\n"
                "*** Add File: src/new.py\n"
                "+value = 1\n"
                "*** End Patch\n"
            ),
        }
        plugin._on_pre_tool_call(
            tool_name="patch",
            args=args,
            session_id="session-1",
            task_id="task-1",
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(
            self.calls[0][0]["will_edit_filepaths"],
            [str(self.repo / "src/new.py")],
        )

    def test_terminal_edit_uses_git_ai_shell_events(self):
        self._seed_session()
        args = {
            "command": "printf generated > src/generated.txt",
            "workdir": str(self.repo),
        }
        plugin._on_pre_tool_call(
            tool_name="terminal",
            args=args,
            session_id="session-1",
            task_id="task-1",
            tool_call_id="shell-call-1",
        )
        plugin._on_post_tool_call(
            tool_name="terminal",
            args=args,
            result="command completed",
            status="ok",
            session_id="session-1",
            task_id="task-1",
            tool_call_id="shell-call-1",
        )

        self.assertEqual(len(self.calls), 2)
        before, after = (payload for payload, _ in self.calls)
        self.assertEqual(before["type"], "pre_shell_command")
        self.assertEqual(after["type"], "post_shell_command")
        self.assertEqual(before["tool_use_id"], "shell-call-1")
        self.assertEqual(after["command"], args["command"])

    def test_failed_file_edit_does_not_get_ai_attribution(self):
        self._seed_session()
        args = {"path": "src/main.py", "content": "bad\n"}
        plugin._on_pre_tool_call(
            tool_name="write_file",
            args=args,
            session_id="session-1",
            task_id="task-1",
        )
        plugin._on_post_tool_call(
            tool_name="write_file",
            args=args,
            result=json.dumps({"error": "denied"}),
            status="error",
            session_id="session-1",
            task_id="task-1",
        )
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][0]["type"], "human")

    def test_missing_git_ai_is_a_noop(self):
        with patch.dict(
            os.environ,
            {"HERMES_GIT_AI_BIN": "", "HERMES_GIT_AI_DISABLED": ""},
        ), patch.object(plugin.shutil, "which", return_value=None):
            self.assertIsNone(plugin._git_ai_command())

    def test_checkpoint_process_receives_agent_v1_stdin(self):
        self.runner.stop()
        try:
            with tempfile.TemporaryDirectory() as directory:
                directory_path = Path(directory)
                executable = directory_path / "git-ai"
                log_path = directory_path / "call.json"
                executable.write_text(
                    "\n".join(
                        [
                            "#!/usr/bin/env python3",
                            "import json, os, sys",
                            "payload = json.load(sys.stdin)",
                            "with open(os.environ['GIT_AI_TEST_LOG'], 'w') as handle:",
                            "    json.dump({'argv': sys.argv[1:], 'payload': payload}, handle)",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                executable.chmod(0o755)
                with patch.dict(
                    os.environ,
                    {
                        "HERMES_GIT_AI_BIN": str(executable),
                        "GIT_AI_TEST_LOG": str(log_path),
                        "HERMES_GIT_AI_TIMEOUT": "5",
                    },
                ):
                    self.assertTrue(
                        plugin._run_checkpoint(
                            {
                                "type": "ai_agent",
                                "repo_working_dir": str(self.repo),
                                "agent_name": "hermes",
                                "model": "gpt-5.6-luna",
                                "conversation_id": "session-1",
                            },
                            self.repo,
                        )
                    )
                recorded = json.loads(log_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    recorded["argv"],
                    ["checkpoint", "agent-v1", "--hook-input", "stdin"],
                )
                self.assertEqual(recorded["payload"]["conversation_id"], "session-1")
        finally:
            self.runner.start()

    def test_transcript_can_be_disabled(self):
        self._seed_session()
        args = {"path": "src/main.py", "content": "print(1)\n"}
        with patch.dict(os.environ, {"HERMES_GIT_AI_TRANSCRIPT": "off"}):
            plugin._on_post_tool_call(
                tool_name="write_file",
                args=args,
                result=json.dumps({"ok": True}),
                status="ok",
                session_id="session-1",
                task_id="task-1",
            )
        self.assertNotIn("transcript", self.calls[0][0])


if __name__ == "__main__":
    unittest.main()
