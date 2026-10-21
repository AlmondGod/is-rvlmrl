"""User-authorized Codex route, explicitly distinct from a raw Astra API call.

Each decision is a fresh CLI session with only public inputs. No simulator imports,
session resume, model-written code execution, or extraction of login credentials.
Disable shell, image-file tools, integrations and memories. Reject any tool event.
The CLI adds its own context; this is a wrapper confound, not identical full prompts.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from .interface import SYSTEM, SKILLS

CLI = Path("/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex")
CLI_SHA256 = "0d5ac7aae20c4eda6242428d5e218c55ba96d8763fda2a271088ff37104b9d28"
DISABLED = ("shell_tool", "view_image", "apps", "plugins", "remote_plugin",
            "skill_search", "multi_agent", "browser_use", "computer_use", "hooks",
            "memories", "image_generation", "goals", "workspace_dependencies",
            "unbounded_connection_retries")

def parse_events(events, max_tokens):
    usage, thread = None, None
    for event in events:
        if event.get("type") == "thread.started":
            thread = event["thread_id"]
        if event.get("type") in ("turn.failed", "error"):
            raise ValueError("codex_turn_failed")
        if event.get("type") in ("item.started", "item.completed", "item.updated"):
            item = event.get("item", {})
            if item.get("type") not in ("agent_message", "reasoning", "error"):
                raise ValueError("codex_disallowed_tool_or_item:" + str(item.get("type")))
            if item.get("type") == "error" and not item.get("message", "").startswith(
                    "Under-development features enabled: skip_host_skill_discovery."):
                raise ValueError("codex_error_item")
        if event.get("type") == "turn.completed":
            usage = event["usage"]
    if usage is None:
        raise ValueError("codex_no_completed_turn")
    # CLI has no documented generation-time max_output_tokens option. This is a
    # post-response gate; retain that distinction from Qwen's generation-time cap.
    if usage.get("output_tokens", 0) > max_tokens:
        raise ValueError("codex_output_token_budget")
    return usage, thread

def query_codex(*, model, packet, images, settings, max_tokens, timeout):
    if model != "gpt-6-astra":
        raise ValueError("This frozen Codex route is for gpt-6-astra only.")
    if hashlib.sha256(CLI.read_bytes()).hexdigest() != CLI_SHA256:
        raise RuntimeError("Codex executable changed since preflight.")
    text = json.dumps({"skills": SKILLS, "observation": packet}, sort_keys=True)
    shared_hash = hashlib.sha256(SYSTEM.encode() + text.encode())
    for name, value in images.items():
        shared_hash.update(name.encode())
        shared_hash.update(value)
    with tempfile.TemporaryDirectory(prefix="assembly-public-") as folder:
        root = Path(folder)
        instructions, answer = root / "instructions.txt", root / "answer.json"
        instructions.write_text(SYSTEM)
        args = [str(CLI), "exec", "--ignore-user-config", "--ignore-rules",
                "--strict-config", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--cd", str(root), "--model", model]
        for key, value in {
            "model_reasoning_effort": settings.get("reasoning_effort", "low"),
            "model_instructions_file": str(instructions),
            "project_doc_max_bytes": 0, "web_search": "disabled"
        }.items():
            args += ["-c", key + "=" + json.dumps(value)]
        for feature in DISABLED:
            args += ["--disable", feature]
        args += ["--enable", "skip_host_skill_discovery", "--json",
                 "--output-last-message", str(answer)]
        for name, value in images.items():
            path = root / (name + ".png")
            path.write_bytes(value)
            args += ["--image", str(path)]
        args += ["-"]
        # Do not inherit the desktop task's tools pipe, thread identity or
        # permission-profile override. Preserve the user's normal HOME/auth.
        child_env = {k: v for k, v in os.environ.items() if not k.startswith("CODEX_")}
        started = time.monotonic()
        result = {"request_sha256": shared_hash.hexdigest(),
                  "request_hash_scope": "shared system, public packet, named image bytes",
                  "backend": "codex_cli", "cli_sha256": CLI_SHA256,
                  "requested_model": model,
                  "model_identity_source": "CLI configuration; no independently returned model snapshot",
                  "generation_token_cap_enforced": False,
                  "postresponse_output_token_gate": max_tokens}
        try:
            process = subprocess.run(args, input=text, text=True, capture_output=True,
                                     env=child_env, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            result.update(error="codex_timeout", latency_wall_s=time.monotonic() - started,
                          partial_stdout=(exc.stdout or b"").decode(errors="replace")
                          if isinstance(exc.stdout, bytes) else exc.stdout)
            return result
        result.update(latency_wall_s=time.monotonic() - started,
                      cli_exit_code=process.returncode, stdout=process.stdout,
                      stderr=process.stderr[-16000:])
        try:
            if process.returncode:
                raise ValueError("codex_nonzero_exit")
            events = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
            usage, thread = parse_events(events, max_tokens)
            result.update(text=answer.read_text(),
                          response={"model": "codex-configured:" + model,
                                    "id": thread, "usage": usage})
        except Exception as exc:
            result["error"] = str(exc)
        return result
