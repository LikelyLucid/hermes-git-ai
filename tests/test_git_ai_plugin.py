from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
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

    def _git_reference(self, directory):
        with tempfile.TemporaryFile(mode="w+t") as output:
            result = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                                    stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.DEVNULL, timeout=15, check=False)
            output.seek(0)
            value = output.read().strip()
        return result.returncode, os.path.normpath(value) if value else None

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
                executable = directory_path / "git-ai.py"
                log_path = directory_path / "call.json"
                executable.write_text(
                    "\n".join(
                        [
                            "import json, os, sys",
                            "payload = json.load(sys.stdin)",
                            "with open(os.environ['GIT_AI_TEST_LOG'], 'w') as handle:",
                            "    json.dump({'argv': sys.argv[1:], 'payload': payload}, handle)",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with patch.dict(
                    os.environ,
                    {
                        "GIT_AI_TEST_LOG": str(log_path),
                        "HERMES_GIT_AI_TIMEOUT": "5",
                    },
                ), patch.object(
                    plugin, "_git_ai_command", return_value=[sys.executable, str(executable)]
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

    def test_git_discovery_uses_file_output_and_fails_open_on_timeout(self):
        def timeout(argv, **kwargs):
            self.assertEqual(kwargs.get("stdin"), subprocess.DEVNULL)
            self.assertEqual(kwargs.get("stderr"), subprocess.DEVNULL)
            self.assertNotEqual(kwargs.get("stdout"), subprocess.PIPE)
            self.assertTrue(hasattr(kwargs.get("stdout"), "fileno"))
            self.assertGreater(kwargs.get("timeout", 0), 0)
            self.assertLessEqual(kwargs["timeout"], 10)
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

        with patch.object(plugin.subprocess, "run", side_effect=timeout) as run:
            self.assertIsNone(plugin._repo_root(self.repo))
            self.assertEqual(run.call_count, 1)
            self.assertIsNone(plugin._repo_root(self.repo))
            self.assertEqual(run.call_count, 2)

    def test_git_ai_executable_override_is_selected(self):
        with patch.dict(os.environ, {"HERMES_GIT_AI_BIN": sys.executable,
                                     "HERMES_GIT_AI_DISABLED": ""}):
            self.assertEqual(plugin._git_ai_command(), [sys.executable])

    def test_discovery_preserves_unicode_roots_with_utf8_mode_off(self):
        env = os.environ.copy()
        env.update(PYTHONUTF8="0", PYTHONIOENCODING="utf-8")
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output:
            child = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()),
                 "--unicode-child", str(self.repo)],
                env=env, stdin=subprocess.DEVNULL, stdout=output,
                stderr=output, timeout=60, check=False,
            )
            output.seek(0)
            details = output.read()
        print(details)
        self.assertEqual(child.returncode, 0, details)

    def test_discovery_caches_only_successful_roots(self):
        self.assertEqual(plugin._repo_root(self.repo), str(self.repo))
        with patch.object(plugin.subprocess, "run", side_effect=OSError("unavailable")) as run:
            self.assertEqual(plugin._repo_root(self.repo), str(self.repo))
            self.assertEqual(run.call_count, 0)
            plugin._REPO_CACHE.clear()
            self.assertIsNone(plugin._repo_root(self.repo))
            self.assertIsNone(plugin._repo_root(self.repo))
            self.assertEqual(run.call_count, 2)

    def test_invalid_markers_follow_git_discovery(self):
        for outer in (False, True):
            for kind in ("file", "directory"):
                with self.subTest(outer=outer, kind=kind):
                    if outer:
                        parent = self.repo
                    else:
                        standalone = tempfile.TemporaryDirectory()
                        self.addCleanup(standalone.cleanup)
                        parent = Path(standalone.name)
                    directory = parent / ("invalid-%s-%s" % (outer, kind))
                    directory.mkdir()
                    marker = directory / ".git"
                    if kind == "file":
                        marker.write_text("invalid marker\n", encoding="utf-8")
                    else:
                        marker.mkdir()
                    plugin._REPO_CACHE.clear()
                    expected = str(self.repo) if outer and kind == "directory" else None
                    reference = self._git_reference(directory)
                    self.assertEqual(reference, (0 if expected else 128, expected))
                    print('invalid marker', outer, kind, 'git', reference)
                    self.assertEqual(plugin._repo_root(directory), expected)
                    self.calls.clear()
                    plugin._on_pre_tool_call(tool_name="write_file", args={"path": str(directory / "new.txt")})
                    self.assertEqual(len(self.calls), int(expected is not None))

    @unittest.skipUnless(os.name == "nt", "Windows inherited handles required")
    def test_discovery_does_not_wait_for_descendant_stdout_close(self):
        directory = Path(self.tempdir.name)
        script = directory / "inherited_stdout.py"
        ready, release, done = (directory / name for name in ("ready", "release", "done"))
        script.write_text(
            "import os, subprocess, sys, time\n"
            "from pathlib import Path\n"
            "base = Path(sys.argv[2])\n"
            "if sys.argv[1] == 'child':\n"
            "    (base / 'ready').write_text(str(os.getpid()))\n"
            "    deadline = time.monotonic() + 30\n"
            "    while not (base / 'release').exists() and time.monotonic() < deadline:\n"
            "        time.sleep(0.05)\n"
            "    os.write(1, b'descendant still owns stdout\\n')\n"
            "    (base / 'done').write_text('done')\n"
            "else:\n"
            "    subprocess.Popen([sys.executable, __file__, 'child', str(base)], close_fds=False)\n"
            "    deadline = time.monotonic() + 20\n"
            "    while not (base / 'ready').exists() and time.monotonic() < deadline:\n"
            "        time.sleep(0.05)\n"
            "    if not (base / 'ready').exists(): sys.exit(3)\n"
            "    print(str(base), flush=True)\n",
            encoding="utf-8",
        )
        real_run = subprocess.run

        def invoke_fixture(argv, **kwargs):
            return real_run([sys.executable, str(script), "parent", str(directory)], **kwargs)

        try:
            with patch.object(plugin.subprocess, "run", side_effect=invoke_fixture):
                root = plugin._repo_root(directory)
            self.assertEqual(root, str(directory))
            self.assertTrue(ready.exists())
            self.assertFalse(done.exists(), "descendant already closed stdout")
            print('inherited stdout: root returned before child release; child pid', ready.read_text())
        finally:
            release.write_text("release", encoding="utf-8")
            deadline = time.monotonic() + 20
            while ready.exists() and not done.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if ready.exists() and not done.exists():
                real_run(["taskkill", "/PID", ready.read_text(), "/T", "/F"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            self.assertTrue(done.exists(), "descendant cleanup did not complete")
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(0x100000, False, int(ready.read_text()))
            if handle:
                try:
                    self.assertEqual(kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 20000), 0)
                finally:
                    kernel32.CloseHandle(ctypes.c_void_p(handle))
            print('inherited stdout: child wrote after release and exited')

    @unittest.skipUnless(os.name == "nt", "Windows junctions required")
    def test_directory_junctions_discover_target_repository(self):
        target = self.repo / "src"
        outside = Path(self.tempdir.name).parent / (self.repo.name + "-alias")
        for alias in (outside, self.repo / "src" / "nested-alias"):
            with self.subTest(alias=str(alias)):
                # The second alias is under a different, valid repository.
                if alias.parent == target:
                    outer_temp = tempfile.TemporaryDirectory()
                    self.addCleanup(outer_temp.cleanup)
                    outer = Path(outer_temp.name)
                    subprocess.run(["git", "init", "-q", str(outer)], check=True)
                    alias = outer / "alias"
                subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(alias), str(target)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                try:
                    self.calls.clear()
                    plugin._REPO_CACHE.clear()
                    reference = self._git_reference(alias)
                    self.assertEqual(reference, (0, str(self.repo)))
                    print('junction', str(alias), 'git', reference)
                    args = {"path": str(alias / "new.txt")}
                    plugin._on_pre_tool_call(tool_name="write_file", args=args)
                    plugin._on_post_tool_call(tool_name="write_file", args=args, status="ok")
                    self.assertEqual([p["type"] for p, _ in self.calls], ["human", "ai_agent"])
                    self.assertEqual([cwd for _, cwd in self.calls], [str(self.repo)] * 2)
                finally:
                    os.rmdir(alias)

    def test_file_edits_find_normal_nested_and_linked_worktree_roots(self):
        nested = self.repo / "nested"
        subprocess.run(["git", "init", "-q", str(nested)], check=True)
        subprocess.run(
            ["git", "-C", str(self.repo), "-c", "user.name=Test User",
             "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
             "commit", "--allow-empty", "-qm", "Initial commit"], check=True,
        )
        worktree = self.repo / "linked"
        subprocess.run(
            ["git", "-C", str(self.repo), "worktree", "add", "-q", "--detach", str(worktree)],
            check=True,
        )
        self.assertTrue((worktree / ".git").is_file())
        for root in (self.repo, nested, worktree):
            with self.subTest(root=root.name):
                self.calls.clear()
                plugin._on_pre_tool_call(
                    tool_name="write_file", args={"path": str(root / "new.txt"), "content": "new\n"},
                    session_id="session-1", task_id="task-1",
                )
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.calls[0][1], str(root))

    def test_edits_outside_a_repository_are_not_checkpointed(self):
        with tempfile.TemporaryDirectory() as directory:
            plugin._on_pre_tool_call(
                tool_name="write_file", args={"path": str(Path(directory) / "new.txt")},
                session_id="session-1", task_id="task-1",
            )
        self.assertEqual(self.calls, [])


def _unicode_child(directory):
    import locale

    class UnicodeRoots(unittest.TestCase):
        def test_roots_and_cached_roots_match_git(self):
            self.assertEqual(sys.flags.utf8_mode, 0)
            print("child module:", plugin.__file__, "UTF-8 mode:", sys.flags.utf8_mode,
                  "locale:", locale.getpreferredencoding(False), flush=True)
            for name in ("ascii_repo", "caf\u00e9_repo", "\u4e2d\u6587_repo"):
                root = Path(directory) / name
                nested = root / "sub"
                nested.mkdir(parents=True)
                subprocess.run(["git", "init", "-q", str(root)], check=True,
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=15)
                with tempfile.TemporaryFile(mode="w+b") as output:
                    reference = subprocess.run(
                        ["git", "-C", str(nested), "rev-parse", "--show-toplevel"],
                        stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.DEVNULL,
                        timeout=15, check=True,
                    )
                    output.seek(0)
                    expected = os.path.normpath(output.read().decode("utf-8").strip())
                self.assertEqual(reference.returncode, 0)
                self.assertEqual(expected, os.path.normpath(str(root)))
                plugin._REPO_CACHE.clear()
                first = plugin._repo_root(nested)
                with patch.object(plugin.subprocess, "run", side_effect=AssertionError("cache missed")):
                    cached = plugin._repo_root(nested)
                print(ascii(name), "expected", ascii(expected), "first", ascii(first),
                      "cached", ascii(cached), flush=True)
                with self.subTest(root=name, result="first"):
                    self.assertEqual(first, expected)
                with self.subTest(root=name, result="cached"):
                    self.assertEqual(cached, expected)

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(UnicodeRoots)
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--unicode-child":
        raise SystemExit(_unicode_child(sys.argv[2]))
    unittest.main()
