# Git AI support for Hermes Agent

This is a native Hermes Agent plugin that emits Git AI's `agent-v1` checkpoint
payloads around Hermes tool calls.

## What it tracks

- `write_file` and `patch`: marks the pre-edit state as human, then the
  resulting paths as `ai_agent`.
- `terminal`: emits Git AI's `pre_shell_command` and `post_shell_command`
  events, so shell-created edits use Git AI's bash tracking path.
- Model name, Hermes session ID, agent name, and a filtered transcript of user,
  assistant, and tool-use messages are included in AI checkpoint payloads.
- Subagent sessions work automatically because Hermes fires the same hooks for
  them.

The plugin is fail-open: a missing or failing `git-ai` never blocks a Hermes
edit. Non-local terminal backends are skipped because their files are not on
this host. Background commands that keep editing after their `terminal` call
returns cannot be attributed until Hermes exposes their completion as a tool
hook.

## Requirements

Install Git AI separately and make `git-ai` available on `PATH`. This plugin
intentionally does not run an installer or add repository hooks for you.

The plugin invokes the equivalent of:

```text
git ai checkpoint agent-v1 --hook-input stdin
```

For the current Git AI preset, shell events are supported directly by the same
`agent-v1` input schema.

## Configuration

Optional environment variables:

- `HERMES_GIT_AI_DISABLED=1` disables the integration.
- `HERMES_GIT_AI_AGENT_NAME=...` changes the Git AI agent name (default:
  `hermes`).
- `HERMES_GIT_AI_TIMEOUT=10` changes the per-checkpoint subprocess timeout in
  seconds.
- `HERMES_GIT_AI_TRANSCRIPT=off` omits the filtered transcript from payloads.
- `HERMES_GIT_AI_BIN=/absolute/path/to/git-ai` overrides PATH discovery.

## Install / validate

The installed copy lives in the active profile's native plugin directory:
`$HERMES_HOME/plugins/git-ai/`.

```bash
git clone https://github.com/LikelyLucid/hermes-git-ai.git \
  "${HERMES_HOME:-$HOME/.hermes}/plugins/git-ai"
hermes plugins enable git-ai --no-allow-tool-override
hermes plugins doctor git-ai
hermes plugins list --enabled --user --plain
```

The plugin needs a Hermes restart/reload boundary after installation so the
agent process discovers its hooks.
