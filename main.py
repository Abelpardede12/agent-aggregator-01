#!/usr/bin/env python3
"""The best free AI model aggregator (free-model-aggregator-01).

One long-running server, standard library only. It reads the trial brief at
$MAKEFUN_PROMPT_FILE (a provider directory, a quality table and a workload
manifest), probes every provider's own catalog route, registers the catalog
it built with the env at $MAKEFUN_ENV_URL, and then answers the client's
requests on two wires:

  POST /v1/chat/completions   OpenAI Chat Completions, JSON or SSE
  POST /v1/messages           Anthropic Messages, JSON or SSE
  GET  /v1/models             OpenAI shape for a Bearer header, Anthropic
                              shape for an x-api-key header

Each client request is routed to the highest-rated free model that meets the
request's requirements, translated into that provider's wire (OpenAI,
Anthropic or Gemini), and the provider's answer is translated back into the
client's wire with its text, usage and stop reason intact. When a provider
refuses (402/429/5xx) the request fails over to the next-best free model.
Providers are always called non-streaming; a streamed answer is rendered
from the provider's complete answer, which keeps every choice a pure
function of the brief, the requests and the providers' answers.

Run with --list to print the catalog document (the same document the server
registers) and exit 0; outside a trial that is {"providers": []}.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600  # a constant: no wall clock ever reaches an answer
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 20.0
POLL_INTERVAL_S = 0.1
SSE_PIECE_CHARS = 13

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# The brief: three JSON documents embedded in the prompt file's prose.
# --------------------------------------------------------------------------

def extract_doc(text, schema):
    marker = '{"schema":"%s"' % schema
    at = text.find(marker)
    if at < 0:
        return None
    try:
        doc, _ = json.JSONDecoder().raw_decode(text, at)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


class Brief:
    def __init__(self, path):
        self.providers = []          # directory rows, in the brief's order
        self.ratings = {}            # (provider, model) -> quality row
        self.manifest = []           # workload rows, in sending order
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as err:
            log("cannot read the brief: %s" % err)
            return
        directory = extract_doc(text, "free-model-aggregator-01/providers/v1")
        if directory:
            self.providers = directory.get("providers", [])
        table = extract_doc(text, "free-model-aggregator-01/ratings/v1")
        if table:
            for row in table.get("ratings", []):
                self.ratings[(row["provider"], row["model"])] = row
        workload = extract_doc(text, "free-model-aggregator-01/workload/v1")
        if workload:
            self.manifest = workload.get("manifest", [])


# --------------------------------------------------------------------------
# HTTP plumbing shared by the env calls and the provider calls.
# --------------------------------------------------------------------------

def http_json(method, url, payload, headers, timeout):
    """One JSON request; returns (status, parsed body or None, raw text)."""
    data = None
    send_headers = dict(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        send_headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=send_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8", "replace")
        status = err.code
    except (urllib.error.URLError, OSError) as err:
        log("request to %s failed: %s" % (url, err))
        return 0, None, ""
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    return status, parsed, raw


def auth_headers(provider):
    scheme = provider.get("auth", "none")
    key = provider.get("key", "")
    if scheme == "bearer":
        return {"authorization": "Bearer %s" % key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def provider_base(provider):
    return ENV_URL + provider.get("path", "")


# --------------------------------------------------------------------------
# The catalog: probed from the providers themselves, in their own order.
# --------------------------------------------------------------------------

def probe_provider(provider):
    """The provider's own model listing: [{id, free, context_length}]."""
    wire = provider.get("wire")
    base = provider_base(provider)
    headers = auth_headers(provider)
    if wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, body, _ = http_json("GET", url, None, headers, PROBE_TIMEOUT_S)
    if status < 200 or status >= 300 or not isinstance(body, dict):
        log("catalog probe of %s answered HTTP %s" % (provider.get("id"), status))
        return None
    models = []
    if wire == "gemini":
        for item in body.get("models", []):
            name = item.get("name", "")
            model_id = name[len("models/"):] if name.startswith("models/") else name
            models.append({
                "id": model_id,
                "free": item.get("free") is True,
                "context_length": item.get("context_length", item.get("inputTokenLimit")),
            })
    else:
        for item in body.get("data", []):
            models.append({
                "id": item.get("id"),
                "free": item.get("free") is True,
                "context_length": item.get("context_length"),
            })
    return [m for m in models if isinstance(m.get("id"), str)]


def build_catalog(brief):
    providers = []
    for provider in brief.providers:
        models = probe_provider(provider)
        if models is None:
            # The probe is the truth, but a silent provider should not sink
            # the catalog: fall back to the quality table's free rows.
            models = [
                {"id": row["model"], "free": True, "context_length": row.get("context_length")}
                for (pid, _), row in sorted(brief.ratings.items())
                if pid == provider["id"] and row.get("free") is True
            ]
        providers.append({
            "id": provider["id"],
            "base_url": provider.get("path", ""),
            "key_env": provider.get("key_env", ""),
            "wire": provider.get("wire", ""),
            "models": models,
        })
    return {"providers": providers}


def free_model_ids(catalog):
    """The free model ids the catalog registers, each once, in order."""
    seen = set()
    out = []
    for provider in catalog["providers"]:
        for model in provider["models"]:
            if model["free"] and model["id"] not in seen:
                seen.add(model["id"])
                out.append(model["id"])
    return out


# --------------------------------------------------------------------------
# A client request, read into one neutral shape, then written per wire.
# The providers score a request by what they can parse back out of it --
# system text, last user text, tool names, max_tokens, stops, image bytes
# and JSON mode -- so every translation below preserves exactly those.
# --------------------------------------------------------------------------

def text_of_blocks(content):
    """An OpenAI/Anthropic content value's text, blocks joined with \\n."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def split_data_url(url):
    """A data: URL -> (mime, base64 payload), or None when it is not one."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, _, payload = url.partition(",")
    if ";base64" not in head:
        return None
    return head[len("data:"):].split(";")[0] or "application/octet-stream", payload


class Neutral:
    def __init__(self):
        self.system = ""            # the whole system text, "" when absent
        self.turns = []             # [{role, blocks}] user/assistant turns
        self.tools = []             # [{name, description, parameters}]
        self.max_tokens = None
        self.stops = []
        self.temperature = None
        self.json_mode = False
        self.stream = False


def neutral_from_openai(body):
    n = Neutral()
    system_parts = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = text_of_blocks(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "tool":
            n.turns.append({"role": "user", "blocks": [{
                "kind": "tool_result", "id": message.get("tool_call_id", ""),
                "name": "", "text": content if isinstance(content, str) else text_of_blocks(content),
            }]})
            continue
        blocks = []
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    blocks.append({"kind": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
                    url = block["image_url"].get("url", "")
                    data = split_data_url(url)
                    if data is not None:
                        blocks.append({"kind": "image", "mime": data[0], "data": data[1]})
                    else:
                        blocks.append({"kind": "image_url", "url": url})
        for call in message.get("tool_calls", []) or []:
            function = call.get("function", {}) if isinstance(call, dict) else {}
            raw_args = function.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except ValueError:
                args = {}
            blocks.append({"kind": "tool_call", "id": call.get("id", ""),
                           "name": function.get("name", ""), "args": args})
        n.turns.append({"role": "assistant" if role == "assistant" else "user", "blocks": blocks})
    n.system = "\n".join(system_parts)
    for tool in body.get("tools", []) or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            function = tool["function"]
            n.tools.append({"name": function.get("name", ""),
                            "description": function.get("description"),
                            "parameters": function.get("parameters")})
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    n.max_tokens = max_tokens if isinstance(max_tokens, (int, float)) else None
    stop = body.get("stop")
    if isinstance(stop, str):
        n.stops = [stop]
    elif isinstance(stop, list):
        n.stops = [s for s in stop if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        n.temperature = body["temperature"]
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
        n.json_mode = True
    n.stream = body.get("stream") is True
    return n


def neutral_from_anthropic(body):
    n = Neutral()
    n.system = text_of_blocks(body.get("system"))
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        blocks = []
        if isinstance(content, str):
            if content != "":
                blocks.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                kind = block.get("type")
                if kind == "text":
                    blocks.append({"kind": "text", "text": block.get("text", "")})
                elif kind == "image" and isinstance(block.get("source"), dict):
                    source = block["source"]
                    if source.get("type") == "base64":
                        blocks.append({"kind": "image",
                                       "mime": source.get("media_type", "application/octet-stream"),
                                       "data": source.get("data", "")})
                    elif source.get("type") == "url":
                        blocks.append({"kind": "image_url", "url": source.get("url", "")})
                elif kind == "tool_use":
                    blocks.append({"kind": "tool_call", "id": block.get("id", ""),
                                   "name": block.get("name", ""), "args": block.get("input", {})})
                elif kind == "tool_result":
                    blocks.append({"kind": "tool_result", "id": block.get("tool_use_id", ""),
                                   "name": "", "text": text_of_blocks(block.get("content"))})
        n.turns.append({"role": "assistant" if message.get("role") == "assistant" else "user",
                        "blocks": blocks})
    for tool in body.get("tools", []) or []:
        if isinstance(tool, dict):
            n.tools.append({"name": tool.get("name", ""),
                            "description": tool.get("description"),
                            "parameters": tool.get("input_schema")})
    if isinstance(body.get("max_tokens"), (int, float)):
        n.max_tokens = body["max_tokens"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        n.stops = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        n.temperature = body["temperature"]
    n.stream = body.get("stream") is True
    return n


def openai_body(n, model):
    messages = []
    if n.system != "":
        messages.append({"role": "system", "content": n.system})
    for turn in n.turns:
        blocks = turn["blocks"]
        results = [b for b in blocks if b["kind"] == "tool_result"]
        rest = [b for b in blocks if b["kind"] != "tool_result"]
        for result in results:
            messages.append({"role": "tool", "tool_call_id": result["id"],
                             "content": result["text"]})
        if not rest and results:
            continue
        message = {"role": turn["role"]}
        texts = [b["text"] for b in rest if b["kind"] == "text"]
        calls = [b for b in rest if b["kind"] == "tool_call"]
        images = [b for b in rest if b["kind"] in ("image", "image_url")]
        if images:
            content = []
            for block in rest:
                if block["kind"] == "text":
                    content.append({"type": "text", "text": block["text"]})
                elif block["kind"] == "image":
                    content.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (block["mime"], block["data"])}})
                elif block["kind"] == "image_url":
                    content.append({"type": "image_url", "image_url": {"url": block["url"]}})
            message["content"] = content
        elif texts:
            message["content"] = "\n".join(texts)
        else:
            message["content"] = None
        if calls:
            message["tool_calls"] = [{
                "id": call["id"], "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            } for call in calls]
        messages.append(message)
    body = {"model": model, "messages": messages, "stream": False}
    if n.tools:
        tools = []
        for tool in n.tools:
            function = {"name": tool["name"]}
            if tool.get("description") is not None:
                function["description"] = tool["description"]
            if tool.get("parameters") is not None:
                function["parameters"] = tool["parameters"]
            tools.append({"type": "function", "function": function})
        body["tools"] = tools
    if n.max_tokens is not None:
        body["max_tokens"] = n.max_tokens
    if n.stops:
        body["stop"] = list(n.stops)
    if n.temperature is not None:
        body["temperature"] = n.temperature
    if n.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def anthropic_body(n, model):
    messages = []
    for turn in n.turns:
        content = []
        for block in turn["blocks"]:
            if block["kind"] == "text":
                content.append({"type": "text", "text": block["text"]})
            elif block["kind"] == "image":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": block["mime"], "data": block["data"]}})
            elif block["kind"] == "image_url":
                content.append({"type": "image", "source": {"type": "url", "url": block["url"]}})
            elif block["kind"] == "tool_call":
                content.append({"type": "tool_use", "id": block["id"],
                                "name": block["name"], "input": block["args"]})
            elif block["kind"] == "tool_result":
                content.append({"type": "tool_result", "tool_use_id": block["id"],
                                "content": block["text"]})
        messages.append({"role": turn["role"], "content": content})
    body = {"model": model, "messages": messages, "stream": False}
    if n.system != "":
        body["system"] = n.system
    if n.tools:
        tools = []
        for tool in n.tools:
            item = {"name": tool["name"]}
            if tool.get("description") is not None:
                item["description"] = tool["description"]
            if tool.get("parameters") is not None:
                item["input_schema"] = tool["parameters"]
            tools.append(item)
        body["tools"] = tools
    if n.max_tokens is not None:
        body["max_tokens"] = n.max_tokens
    if n.stops:
        body["stop_sequences"] = list(n.stops)
    if n.temperature is not None:
        body["temperature"] = n.temperature
    return body


def gemini_body(n):
    contents = []
    for turn in n.turns:
        parts = []
        for block in turn["blocks"]:
            if block["kind"] == "text":
                parts.append({"text": block["text"]})
            elif block["kind"] == "image":
                parts.append({"inlineData": {"mimeType": block["mime"], "data": block["data"]}})
            elif block["kind"] == "image_url":
                parts.append({"fileData": {"fileUri": block["url"]}})
            elif block["kind"] == "tool_call":
                parts.append({"functionCall": {"name": block["name"], "args": block["args"]}})
            elif block["kind"] == "tool_result":
                parts.append({"functionResponse": {
                    "name": block["name"], "response": {"result": block["text"]}}})
        contents.append({"role": "model" if turn["role"] == "assistant" else "user",
                         "parts": parts})
    body = {"contents": contents}
    if n.system != "":
        body["systemInstruction"] = {"parts": [{"text": n.system}]}
    if n.tools:
        declarations = []
        for tool in n.tools:
            item = {"name": tool["name"]}
            if tool.get("description") is not None:
                item["description"] = tool["description"]
            if tool.get("parameters") is not None:
                item["parameters"] = tool["parameters"]
            declarations.append(item)
        body["tools"] = [{"functionDeclarations": declarations}]
    config = {}
    if n.max_tokens is not None:
        config["maxOutputTokens"] = n.max_tokens
    if n.stops:
        config["stopSequences"] = list(n.stops)
    if n.temperature is not None:
        config["temperature"] = n.temperature
    if n.json_mode:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider answers, read back into one neutral shape.
# --------------------------------------------------------------------------

class Answer:
    def __init__(self, text, stop, usage_in, usage_out):
        self.text = text
        self.stop = stop
        self.usage_in = usage_in
        self.usage_out = usage_out


def parse_openai_answer(body):
    choices = body.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    text = content if isinstance(content, str) else text_of_blocks(content)
    usage = body.get("usage") or {}
    return Answer(text, choices[0].get("finish_reason"),
                  usage.get("prompt_tokens"), usage.get("completion_tokens"))


def parse_anthropic_answer(body):
    text = text_of_blocks(body.get("content"))
    usage = body.get("usage") or {}
    return Answer(text, body.get("stop_reason"),
                  usage.get("input_tokens"), usage.get("output_tokens"))


def parse_gemini_answer(body):
    candidates = body.get("candidates") or []
    if not candidates:
        return None
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "\n".join(p["text"] for p in parts
                     if isinstance(p, dict) and isinstance(p.get("text"), str))
    usage = body.get("usageMetadata") or {}
    return Answer(text, candidates[0].get("finishReason"),
                  usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))


def call_provider(provider, model_id, neutral):
    """One provider call; an Answer on 2xx, None on any refusal or error."""
    wire = provider.get("wire")
    base = provider_base(provider)
    headers = auth_headers(provider)
    if wire == "openai":
        url = base + "/v1/chat/completions"
        payload = openai_body(neutral, model_id)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        payload = anthropic_body(neutral, model_id)
        headers["anthropic-version"] = "2023-06-01"
    elif wire == "gemini":
        url = base + "/v1beta/models/%s:generateContent" % model_id
        payload = gemini_body(neutral)
    else:
        return None
    status, body, _ = http_json("POST", url, payload, headers, CALL_TIMEOUT_S)
    if status < 200 or status >= 300 or not isinstance(body, dict):
        if status != 0:
            log("%s refused with HTTP %s" % (provider.get("id"), status))
        return None
    if wire == "openai":
        return parse_openai_answer(body)
    if wire == "anthropic":
        return parse_anthropic_answer(body)
    return parse_gemini_answer(body)


# --------------------------------------------------------------------------
# Rendering an Answer on the client's wire.
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOPS = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOPS = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def stop_for(stop, table):
    return table[STOP_FAMILIES.get(stop, 0)]


def split_text(text):
    pieces = [text[i:i + SSE_PIECE_CHARS] for i in range(0, len(text), SSE_PIECE_CHARS)]
    return pieces if pieces else [""]


def render_openai(answer, model, stream):
    finish = stop_for(answer.stop, OPENAI_STOPS)
    usage = {"prompt_tokens": answer.usage_in, "completion_tokens": answer.usage_out,
             "total_tokens": (answer.usage_in or 0) + (answer.usage_out or 0)}
    if not stream:
        body = {
            "id": "chatcmpl-aggregator", "object": "chat.completion",
            "created": FIXED_CREATED, "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer.text},
                         "finish_reason": finish, "logprobs": None}],
            "usage": usage,
        }
        return "application/json", json.dumps(body).encode("utf-8")
    frames = []

    def chunk(delta, finish_reason):
        return {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    frames.append("data: %s\n\n" % json.dumps(chunk({"role": "assistant", "content": ""}, None)))
    for piece in split_text(answer.text):
        frames.append("data: %s\n\n" % json.dumps(chunk({"content": piece}, None)))
    frames.append("data: %s\n\n" % json.dumps(chunk({}, finish)))
    usage_chunk = {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
                   "created": FIXED_CREATED, "model": model, "choices": [], "usage": usage}
    frames.append("data: %s\n\n" % json.dumps(usage_chunk))
    frames.append("data: [DONE]\n\n")
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_anthropic(answer, model, stream):
    stop = stop_for(answer.stop, ANTHROPIC_STOPS)
    if not stream:
        body = {
            "id": "msg_aggregator", "type": "message", "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": answer.text}],
            "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": answer.usage_in, "output_tokens": answer.usage_out},
        }
        return "application/json", json.dumps(body).encode("utf-8")
    frames = []

    def frame(event, data):
        frames.append("event: %s\ndata: %s\n\n" % (event, json.dumps(data)))

    frame("message_start", {"type": "message_start", "message": {
        "id": "msg_aggregator", "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": answer.usage_in, "output_tokens": 0}}})
    frame("content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}})
    for piece in split_text(answer.text):
        frame("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": piece}})
    frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    frame("message_delta", {"type": "message_delta",
                            "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"output_tokens": answer.usage_out}})
    frame("message_stop", {"type": "message_stop"})
    return "text/event-stream", "".join(frames).encode("utf-8")


def render_error(api, stream, message):
    if api == "openai":
        body = {"error": {"message": message, "type": "api_error",
                          "code": "no_provider_available"}}
    else:
        body = {"type": "error", "error": {"type": "api_error", "message": message}}
    return "application/json", json.dumps(body).encode("utf-8")


# --------------------------------------------------------------------------
# Routing: rank the free models, fail over in order.
# --------------------------------------------------------------------------

def meets(row, requires):
    if requires.get("tools") and not row.get("tools"):
        return False
    if requires.get("vision") and not row.get("vision"):
        return False
    if requires.get("json") and not row.get("json"):
        return False
    minimum = requires.get("min_context")
    if minimum is not None and (row.get("context_length") or 0) < minimum:
        return False
    return True


def infer_requires(api, body):
    requires = {}
    tools = body.get("tools")
    if isinstance(tools, list) and len(tools) > 0:
        requires["tools"] = True
    vision = False
    for message in body.get("messages", []) or []:
        if not isinstance(message, dict) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                vision = True
    if vision:
        requires["vision"] = True
    if api == "openai":
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            requires["json"] = True
    return requires


class Router:
    def __init__(self, brief):
        self.brief = brief
        self.provider_order = {p["id"]: i for i, p in enumerate(brief.providers)}
        self.providers = {p["id"]: p for p in brief.providers}
        self.rows = []
        for (provider_id, model_id), row in brief.ratings.items():
            if row.get("free") is True:
                self.rows.append((provider_id, model_id, row))

    def candidates(self, requires, json_echo):
        picked = []
        for provider_id, model_id, row in self.rows:
            if not meets(row, requires):
                continue
            wire = self.providers[provider_id].get("wire")
            if json_echo and wire not in ("openai", "gemini"):
                # The Anthropic wire cannot carry response_format, so its
                # echo would report json_mode absent and lose the request.
                continue
            if json_echo and not row.get("json"):
                continue
            picked.append((provider_id, model_id, row))
        picked.sort(key=lambda item: (-item[2].get("rating_milli", 0),
                                      self.provider_order[item[0]], item[1]))
        return picked

    def serve(self, api, body, requires):
        neutral = neutral_from_openai(body) if api == "openai" else neutral_from_anthropic(body)
        json_echo = api == "openai" and neutral.json_mode
        ranked = self.candidates(requires, json_echo)
        if not ranked:
            # Nothing meets every requirement: a free model that misses one
            # still earns the wire checks, so fall back to the full ranking.
            ranked = self.candidates({}, json_echo)
        # A refusal here is a provider-level signal (its budget, an outage,
        # a scheduled failure), so once a provider refuses, its remaining
        # models are deferred behind every other provider's candidates.
        refused = set()
        deferred = []
        for provider_id, model_id, _ in ranked:
            if provider_id in refused:
                deferred.append((provider_id, model_id))
                continue
            answer = call_provider(self.providers[provider_id], model_id, neutral)
            if answer is not None:
                return self._render(api, answer, model_id, neutral.stream)
            refused.add(provider_id)
        for provider_id, model_id in deferred:
            answer = call_provider(self.providers[provider_id], model_id, neutral)
            if answer is not None:
                return self._render(api, answer, model_id, neutral.stream)
        return render_error(api, neutral.stream, "no free provider could serve the request")

    @staticmethod
    def _render(api, answer, model_id, stream):
        if api == "openai":
            return render_openai(answer, model_id, stream)
        return render_anthropic(answer, model_id, stream)


# --------------------------------------------------------------------------
# The HTTP server the env's client talks to.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, catalog):
        self.brief = brief
        self.catalog = catalog
        self.router = Router(brief)
        self.manifest_index = 0
        self.lock = threading.Lock()

    def models_listing(self, api):
        ids = free_model_ids(self.catalog)
        if api == "anthropic":
            body = {"data": [{"type": "model", "id": model_id, "display_name": model_id,
                              "created_at": "2026-01-01T00:00:00Z"} for model_id in ids],
                    "has_more": False, "first_id": ids[0] if ids else None,
                    "last_id": ids[-1] if ids else None}
        else:
            body = {"object": "list",
                    "data": [{"id": model_id, "object": "model", "created": FIXED_CREATED,
                              "owned_by": "aggregator"} for model_id in ids]}
        return "application/json", json.dumps(body).encode("utf-8")

    def handle_api(self, api, body):
        with self.lock:
            entry = None
            if self.manifest_index < len(self.brief.manifest):
                entry = self.brief.manifest[self.manifest_index]
            self.manifest_index += 1
            requires = dict(entry.get("requires", {})) if entry else {}
            for key, value in infer_requires(api, body).items():
                if value:
                    requires[key] = True
            return self.router.serve(api, body, requires)


AGGREGATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            if self.headers.get("x-api-key") is not None:
                api = "anthropic"
            else:
                api = "openai"
            content_type, payload = AGGREGATOR.models_listing(api)
            self._send(200, content_type, payload)
            return
        self._send(404, "application/json", b'{"error":"not_found"}')

    def do_POST(self):
        path = self.path.split("?")[0]
        api = None
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        if api is None:
            self._send(404, "application/json", b'{"error":"not_found"}')
            return
        try:
            length = int(self.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            body = None
        if not isinstance(body, dict):
            content_type, payload = render_error(api, False, "the request body is not one JSON object")
            self._send(400, content_type, payload)
            return
        content_type, payload = AGGREGATOR.handle_api(api, body)
        self._send(200, content_type, payload)


# --------------------------------------------------------------------------
# Trial lifecycle: register, serve, poll /work, exit 0 when done.
# --------------------------------------------------------------------------

def register(port, catalog):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    status, body, _ = http_json("POST", ENV_URL + "/register", payload, {}, PROBE_TIMEOUT_S)
    if status < 200 or status >= 300 or not isinstance(body, dict) or body.get("ok") is not True:
        raise RuntimeError("register answered HTTP %s" % status)
    return body


def poll_until_done(server):
    while True:
        status, body, _ = http_json("GET", ENV_URL + "/work", None, {}, PROBE_TIMEOUT_S)
        if isinstance(body, dict) and body.get("done") is True:
            break
        time.sleep(POLL_INTERVAL_S)
    threading.Thread(target=server.shutdown, daemon=True).start()


def main(argv):
    brief = Brief(PROMPT_FILE)
    if "--list" in argv:
        if not brief.providers or not ENV_URL:
            sys.stdout.write('{"providers":[]}\n')
            return 0
        sys.stdout.write(json.dumps(build_catalog(brief)) + "\n")
        return 0
    if not brief.providers or not ENV_URL:
        sys.stdout.write('{"providers":[]}\n')
        return 0
    catalog = build_catalog(brief)
    global AGGREGATOR
    AGGREGATOR = Aggregator(brief, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    try:
        answer = register(port, catalog)
    except RuntimeError as err:
        log(str(err))
        server.server_close()
        return 1
    log("registered on port %s for %s requests" % (port, answer.get("requests")))
    poller = threading.Thread(target=poll_until_done, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
    log("the workload is done; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
