"""Tool-free model transport, receiving only the serialized public request.

Credentials are read from environment; never serialized. No model text executes.
HTTP responses/errors are saved by the runner, excluding authorization headers.
"""
import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from .interface import SYSTEM, SKILLS

def image_data(image_bytes):
    return "data:image/png;base64," + base64.b64encode(image_bytes).decode()

def request_payload(model, backend, packet, images, settings, max_tokens):
    text = json.dumps({"skills": SKILLS, "observation": packet}, sort_keys=True)
    if backend == "responses":
        content = [{"type": "input_text", "text": text}]
        for name, data in images.items():
            content += [{"type": "input_text", "text": name},
                        {"type": "input_image", "image_url": image_data(data), "detail": "high"}]
        return {"model": model, "instructions": SYSTEM,
                "input": [{"role": "user", "content": content}],
                "max_output_tokens": max_tokens,
                "reasoning": {"effort": settings.get("reasoning_effort", "low")},
                "store": False}
    if backend != "chat":
        raise ValueError("unsupported_backend")
    content = [{"type": "text", "text": text}]
    for name, data in images.items():
        content += [{"type": "text", "text": name},
                    {"type": "image_url", "image_url": {"url": image_data(data)}}]
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM},
               {"role": "user", "content": content}], "max_tokens": max_tokens}
    if "temperature" in settings:
        payload.update(temperature=settings["temperature"], top_p=settings["top_p"])
        payload["chat_template_kwargs"] = {"enable_thinking": settings["enable_thinking"]}
        for key in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
            if key in settings:
                payload[key] = settings[key]
    else:
        payload["reasoning_effort"] = settings.get("reasoning_effort", "low")
    return payload

def extract_text(response, backend):
    if backend == "chat":
        choice = response["choices"][0]
        if choice.get("finish_reason") not in ("stop", None):
            raise ValueError("nonfinal_model_response:" + str(choice.get("finish_reason")))
        return choice["message"]["content"]
    if response.get("status") not in ("completed", None):
        raise ValueError("nonfinal_model_response:" + str(response.get("status")))
    return "".join(part["text"] for item in response.get("output", [])
                   if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")

def query(*, model, backend, base_url, api_key_env, packet, images, settings,
          max_tokens, timeout):
    if backend == "codex":
        from .codex_provider import query_codex
        return query_codex(model=model, packet=packet, images=images, settings=settings,
                           max_tokens=max_tokens, timeout=timeout)
    payload = request_payload(model, backend, packet, images, settings, max_tokens)
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError("missing_credential_environment_variable:" + api_key_env)
        headers["Authorization"] = "Bearer " + key
    endpoint = base_url.rstrip("/") + ("/responses" if backend == "responses" else "/chat/completions")
    started = time.monotonic()
    request = urllib.request.Request(endpoint, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            response = json.load(handle)
    except urllib.error.HTTPError as exc:
        # Error bodies may contain provider details; no credentials or headers.
        return {"error": "http_" + str(exc.code), "provider_error": exc.read().decode()[:4000],
                "latency_wall_s": time.monotonic() - started,
                "request_sha256": hashlib.sha256(body).hexdigest()}
    elapsed = time.monotonic() - started
    result = {"response": response, "latency_wall_s": elapsed,
              "request_sha256": hashlib.sha256(body).hexdigest()}
    try:
        result["text"] = extract_text(response, backend)
    except Exception as exc:
        result["error"] = str(exc)
    return result
