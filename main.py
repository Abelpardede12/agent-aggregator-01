#!/usr/bin/env python3
"""free-model-aggregator-01: one aggregator server over ten simulated free-tier providers.

The server reads a brief (provider directory, quality table, workload manifest), probes
each provider's own catalog route in that provider's wire, registers the catalog it
built with the env, then answers the client's OpenAI Chat Completions and Anthropic
Messages requests -- JSON and server-sent events -- by routing each one to the
highest-rated free model that meets its requirements, failing over in rating order
when a provider refuses. Providers are always called non-streaming; a streamed
client answer is synthesized from the provider's whole answer in the client's own
event shapes. Everything is a pure function of the brief, the requests and the
providers' answers: no clock, no randomness, so a replay reproduces the trial.
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
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 10.0
MAX_ATTEMPTS = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
SSE_PIECE_CHARS = 13

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# One opener for every outbound call; an empty proxy map keeps loopback dials direct.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def is_record(value):
    return isinstance(value, dict)


# ---------------------------------------------------------------------------
# The brief: three JSON documents found by their schema member.
# ---------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object that opens at start, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents(text):
    """Every JSON object in text that carries a string schema, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _string(value):
    return value if isinstance(value, str) else ""


def _number_or_none(value):
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _bool_or_none(value):
    return value if isinstance(value, bool) else None


class Provider:
    def __init__(self, row):
        self.id = _string(row.get("id"))
        self.name = _string(row.get("name")) or self.id
        self.wire = row.get("wire")
        self.auth = row.get("auth") if row.get("auth") in (
            "bearer", "x-api-key", "x-goog-api-key", "none") else "bearer"
        base = ""
        for key in ("path", "base_path", "base_url", "base"):
            if _string(row.get(key)):
                base = _string(row.get(key))
                break
        self.base = base
        self.key_env = _string(row.get("key_env"))
        self.key = _string(row.get("key"))


class Brief:
    def __init__(self, path):
        self.providers = []
        self.ratings = {}       # (provider id, model id) -> quality row
        self.manifest = []      # workload rows in arrival order
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except OSError as err:
            self.error = "cannot read %s: %s" % (path, err)
            return
        docs = _documents(text)
        directory = next((d for d in docs if d.get("schema") == PROVIDERS_SCHEMA
                          and isinstance(d.get("providers"), list)), None)
        if directory is None:
            self.error = "the brief holds no %s document" % PROVIDERS_SCHEMA
            return
        for row in directory["providers"]:
            if is_record(row) and _string(row.get("id")) and \
                    row.get("wire") in ("openai", "anthropic", "gemini"):
                self.providers.append(Provider(row))
        ratings_doc = next((d for d in docs if d.get("schema") == RATINGS_SCHEMA
                            and isinstance(d.get("ratings"), list)), None)
        if ratings_doc is not None:
            for row in ratings_doc["ratings"]:
                if not is_record(row):
                    continue
                model, provider = _string(row.get("model")), _string(row.get("provider"))
                if not model or not provider:
                    continue
                self.ratings[(provider, model)] = {
                    "rating_milli": _number_or_none(row.get("rating_milli")) or 0,
                    "context_length": _number_or_none(row.get("context_length")),
                    "tools": row.get("tools") is True,
                    "vision": row.get("vision") is True,
                    "json": row.get("json") is True,
                    "free": row.get("free") is True,
                }
        workload_doc = next((d for d in docs if d.get("schema") == WORKLOAD_SCHEMA
                             and isinstance(d.get("manifest"), list)), None)
        if workload_doc is not None:
            for row in workload_doc["manifest"]:
                if not is_record(row) or not _string(row.get("id")):
                    continue
                requires = row.get("requires") if is_record(row.get("requires")) else {}
                entry = {
                    "tools": requires.get("tools") is True,
                    "vision": requires.get("vision") is True,
                    "json": requires.get("json") is True,
                }
                if _number_or_none(requires.get("min_context")) is not None:
                    entry["min_context"] = requires["min_context"]
                self.manifest.append(entry)


def join_base(env_url, base):
    """A provider base path joined to the env URL; an absolute base stands as-is."""
    base = (base or "").strip()
    if base.lower().startswith(("http://", "https://")):
        return base.rstrip("/")
    if not base:
        return env_url
    return env_url + ("" if base.startswith("/") else "/") + base.rstrip("/")


def auth_headers(provider):
    if not provider.key:
        return {}
    if provider.auth == "bearer":
        return {"Authorization": "Bearer %s" % provider.key}
    if provider.auth == "x-api-key":
        return {"x-api-key": provider.key}
    if provider.auth == "x-goog-api-key":
        return {"x-goog-api-key": provider.key}
    return {}


def http_call(method, url, headers=None, body=None, timeout=CALL_TIMEOUT_S):
    """One HTTP call; returns (status, parsed JSON or None, raw text)."""
    data = None
    final_headers = dict(headers or {})
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        final_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=final_headers, method=method)
    try:
        with OPENER.open(request, timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as err:
        text = err.read().decode("utf-8", "replace")
        status = err.code
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    return status, parsed, text


# ---------------------------------------------------------------------------
# The catalog probe: each provider's own listing, in its own wire.
# ---------------------------------------------------------------------------

class ProbedModel:
    def __init__(self, model_id, display_name, free, context_length, tools, vision, json_ok):
        self.id = model_id
        self.display_name = display_name
        self.free = free
        self.context_length = context_length
        self.tools = tools
        self.vision = vision
        self.json = json_ok


def _zero_priced(row):
    pricing = row.get("pricing")
    if not is_record(pricing):
        return None
    values = []
    for value in pricing.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, str) and value.strip():
            try:
                values.append(float(value))
            except ValueError:
                pass
    if not values:
        return None
    return all(v == 0 for v in values)


def _model_in(wire, row):
    if not is_record(row):
        return None
    if wire == "gemini":
        model_id = _string(row.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = _string(row.get("id"))
    if not model_id:
        return None
    context = _number_or_none(row.get("context_length"))
    if context is None:
        context = _number_or_none(row.get("context_window"))
    if context is None:
        context = _number_or_none(row.get("inputTokenLimit"))
    free = _bool_or_none(row.get("free"))
    if free is None:
        free = _zero_priced(row)
    display = _string(row.get("display_name")) or _string(row.get("displayName")) or model_id
    return ProbedModel(model_id, display, free, context,
                       _bool_or_none(row.get("tools")), _bool_or_none(row.get("vision")),
                       _bool_or_none(row.get("json")))


class Probe:
    def __init__(self, provider):
        self.provider = provider
        self.ok = False
        self.models = []


def probe_provider(provider):
    probe = Probe(provider)
    base = join_base(ENV_URL, provider.base)
    url = base + ("/v1beta/models" if provider.wire == "gemini" else "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    status, body, _ = http_call("GET", url, headers=headers, timeout=PROBE_TIMEOUT_S)
    if status < 200 or status >= 300 or not is_record(body):
        return probe
    listing = body.get("models") if provider.wire == "gemini" else body.get("data")
    if not isinstance(listing, list):
        return probe
    probe.ok = True
    for row in listing:
        model = _model_in(provider.wire, row)
        if model is not None:
            probe.models.append(model)
    return probe


def model_free(brief, provider_id, model):
    """Free per the quality table where it has a row, else the listing's own word."""
    row = brief.ratings.get((provider_id, model.id))
    if row is not None:
        return row["free"]
    return model.free if model.free is not None else False


def model_caps(brief, provider_id, model):
    row = brief.ratings.get((provider_id, model.id))
    if row is not None:
        return row["context_length"], row["tools"], row["vision"], row["json"]
    return model.context_length, model.tools, model.vision, model.json


def build_catalog(brief, probes):
    """The one catalog document: --list's output and the register body, the same object.

    Providers keep the brief's order; a provider's models keep its listing's order
    (the scored free set is read in that order), first occurrence of an id winning.
    Its base_url is the directory's own path, never joined to the env's origin.
    """
    providers = []
    for probe in probes:
        if not probe.ok:
            continue
        seen = set()
        models = []
        for model in probe.models:
            if model.id in seen:
                continue
            seen.add(model.id)
            row = brief.ratings.get((probe.provider.id, model.id))
            context = row["context_length"] if row is not None else model.context_length
            models.append({
                "id": model.id,
                "free": model_free(brief, probe.provider.id, model),
                "context_length": context,
            })
        providers.append({
            "id": probe.provider.id,
            "base_url": probe.provider.base,
            "key_env": probe.provider.key_env,
            "wire": probe.provider.wire,
            "models": models,
        })
    return {"providers": providers}


# ---------------------------------------------------------------------------
# The neutral form: what a request carries, whichever wire it arrived on.
# The field extraction mirrors the trial's own parsing of each wire, so the
# provider's echo of a translated request matches the client's original.
# ---------------------------------------------------------------------------

def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


class Neutral:
    def __init__(self):
        self.system_text = ""
        self.user_text = ""
        self.image = None          # ("b64", media_type, data) or ("url", url)
        self.tools = []            # {"name", "description", "parameters"}
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.response_format = None
        self.temperature = None
        self.top_p = None


def _tool_neutral(name, description, parameters):
    return {
        "name": name,
        "description": description if isinstance(description, str) else "",
        "parameters": parameters if is_record(parameters) else {"type": "object", "properties": {}},
    }


def _image_from_url(url):
    if url.startswith("data:") and ";base64," in url:
        media, data = url[len("data:"):].split(";base64,", 1)
        return ("b64", media or "image/png", data)
    return ("url", url)


def neutral_from_openai(body):
    n = Neutral()
    messages = [m for m in body.get("messages", []) if is_record(m)] \
        if isinstance(body.get("messages"), list) else []
    system_parts = []
    for message in messages:
        if message.get("role") == "system":
            text = blocks_text(message.get("content"))
            if text:
                system_parts.append(text)
        elif message.get("role") == "user":
            n.user_text = blocks_text(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not is_record(block) or block.get("type") != "image_url":
                        continue
                    image_url = block.get("image_url")
                    if is_record(image_url) and isinstance(image_url.get("url"), str):
                        n.image = _image_from_url(image_url["url"])
    n.system_text = "\n".join(system_parts)
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool) and is_record(tool.get("function")) \
                    and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                n.tools.append(_tool_neutral(fn["name"], fn.get("description"), fn.get("parameters")))
    n.max_tokens = _number_or_none(body.get("max_tokens"))
    if n.max_tokens is None:
        n.max_tokens = _number_or_none(body.get("max_completion_tokens"))
    n.stops = string_list(body.get("stop"))
    response_format = body.get("response_format")
    if is_record(response_format) and response_format.get("type") in ("json_object", "json_schema"):
        n.json_mode = True
        n.response_format = response_format
    n.temperature = _number_or_none(body.get("temperature"))
    n.top_p = _number_or_none(body.get("top_p"))
    return n


def neutral_from_anthropic(body):
    n = Neutral()
    n.system_text = blocks_text(body.get("system"))
    messages = [m for m in body.get("messages", []) if is_record(m)] \
        if isinstance(body.get("messages"), list) else []
    for message in messages:
        if message.get("role") != "user":
            continue
        n.user_text = blocks_text(message.get("content"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not is_record(block) or block.get("type") != "image":
                    continue
                source = block.get("source")
                if not is_record(source):
                    continue
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    media = source.get("media_type")
                    n.image = ("b64", media if isinstance(media, str) else "image/png",
                               source["data"])
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    n.image = ("url", source["url"])
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool) and isinstance(tool.get("name"), str):
                n.tools.append(_tool_neutral(tool["name"], tool.get("description"),
                                              tool.get("input_schema")))
    n.max_tokens = _number_or_none(body.get("max_tokens"))
    n.stops = string_list(body.get("stop_sequences"))
    n.temperature = _number_or_none(body.get("temperature"))
    n.top_p = _number_or_none(body.get("top_p"))
    return n


def requirements_from_neutral(n):
    return {
        "tools": len(n.tools) > 0,
        "vision": n.image is not None,
        "json": n.json_mode,
    }


def upstream_body(wire, model_id, n):
    """The neutral request written on one provider's wire, non-streaming."""
    if wire == "openai":
        messages = []
        if n.system_text:
            messages.append({"role": "system", "content": n.system_text})
        if n.image is None:
            messages.append({"role": "user", "content": n.user_text})
        else:
            parts = [{"type": "text", "text": n.user_text}]
            if n.image[0] == "b64":
                url = "data:%s;base64,%s" % (n.image[1], n.image[2])
            else:
                url = n.image[1]
            parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": parts})
        body = {"model": model_id, "messages": messages}
        if n.max_tokens is not None:
            body["max_tokens"] = n.max_tokens
        if n.stops:
            body["stop"] = list(n.stops)
        if n.tools:
            body["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"],
                "parameters": t["parameters"]}} for t in n.tools]
        if n.json_mode and n.response_format is not None:
            body["response_format"] = n.response_format
        if n.temperature is not None:
            body["temperature"] = n.temperature
        if n.top_p is not None:
            body["top_p"] = n.top_p
        return body
    if wire == "anthropic":
        if n.image is None:
            content = n.user_text
        else:
            if n.image[0] == "b64":
                block = {"type": "image", "source": {
                    "type": "base64", "media_type": n.image[1], "data": n.image[2]}}
            else:
                block = {"type": "image", "source": {"type": "url", "url": n.image[1]}}
            content = [{"type": "text", "text": n.user_text}, block]
        body = {"model": model_id, "messages": [{"role": "user", "content": content}]}
        if n.system_text:
            body["system"] = n.system_text
        if n.max_tokens is not None:
            body["max_tokens"] = n.max_tokens
        if n.stops:
            body["stop_sequences"] = list(n.stops)
        if n.tools:
            body["tools"] = [{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in n.tools]
        if n.temperature is not None:
            body["temperature"] = n.temperature
        if n.top_p is not None:
            body["top_p"] = n.top_p
        return body
    # gemini
    parts = [{"text": n.user_text}]
    if n.image is not None:
        if n.image[0] == "b64":
            parts.append({"inlineData": {"mimeType": n.image[1], "data": n.image[2]}})
        else:
            parts.append({"fileData": {"fileUri": n.image[1]}})
    body = {"contents": [{"role": "user", "parts": parts}]}
    if n.system_text:
        body["systemInstruction"] = {"parts": [{"text": n.system_text}]}
    if n.tools:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]} for t in n.tools]}]
    config = {}
    if n.max_tokens is not None:
        config["maxOutputTokens"] = n.max_tokens
    if n.stops:
        config["stopSequences"] = list(n.stops)
    if n.json_mode:
        config["responseMimeType"] = "application/json"
    if n.temperature is not None:
        config["temperature"] = n.temperature
    if n.top_p is not None:
        config["topP"] = n.top_p
    if config:
        body["generationConfig"] = config
    return body


def upstream_url(provider, model_id):
    base = join_base(ENV_URL, provider.base)
    if provider.wire == "openai":
        return base + "/v1/chat/completions"
    if provider.wire == "anthropic":
        return base + "/v1/messages"
    return "%s/v1beta/models/%s:generateContent" % (base, model_id)


def upstream_headers(provider):
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider))
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


# ---------------------------------------------------------------------------
# Provider answers, read back into one shape: text, usage, native stop.
# ---------------------------------------------------------------------------

def parse_answer(wire, body):
    if not is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            return None
        message = choices[0].get("message")
        if not is_record(message):
            return None
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(b.get("text") for b in content
                           if is_record(b) and b.get("type") == "text"
                           and isinstance(b.get("text"), str))
        else:
            text = ""
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {"text": text,
                "input": _number_or_none(usage.get("prompt_tokens")) or 0,
                "output": _number_or_none(usage.get("completion_tokens")) or 0,
                "stop": choices[0].get("finish_reason")}
    if wire == "anthropic":
        content = body.get("content")
        text = ""
        if isinstance(content, list):
            text = "".join(b.get("text") for b in content
                           if is_record(b) and b.get("type") == "text"
                           and isinstance(b.get("text"), str))
        usage = body.get("usage") if is_record(body.get("usage")) else {}
        return {"text": text,
                "input": _number_or_none(usage.get("input_tokens")) or 0,
                "output": _number_or_none(usage.get("output_tokens")) or 0,
                "stop": body.get("stop_reason")}
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content") if is_record(candidates[0].get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p.get("text") for p in parts
                   if is_record(p) and isinstance(p.get("text"), str))
    usage = body.get("usageMetadata") if is_record(body.get("usageMetadata")) else {}
    return {"text": text,
            "input": _number_or_none(usage.get("promptTokenCount")) or 0,
            "output": _number_or_none(usage.get("candidatesTokenCount")) or 0,
            "stop": candidates[0].get("finishReason")}


# Stop reasons move between wires inside their families: normal, length, tools, refusal.
_TO_OPENAI_STOP = {
    "stop": "stop", "end_turn": "stop", "STOP": "stop",
    "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
    "tool_calls": "tool_calls", "function_call": "tool_calls", "tool_use": "tool_calls",
    "content_filter": "content_filter", "refusal": "content_filter", "SAFETY": "content_filter",
}
_TO_ANTHROPIC_STOP = {
    "stop": "end_turn", "end_turn": "end_turn", "STOP": "end_turn",
    "length": "max_tokens", "max_tokens": "max_tokens", "MAX_TOKENS": "max_tokens",
    "tool_calls": "tool_use", "function_call": "tool_use", "tool_use": "tool_use",
    "content_filter": "refusal", "refusal": "refusal", "SAFETY": "refusal",
}


# ---------------------------------------------------------------------------
# The aggregator: routing, failover, and the client answers.
# ---------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, probes, catalog):
        self.brief = brief
        self.probes = {p.provider.id: p for p in probes if p.ok}
        self.catalog = catalog
        self.deferred = set()   # provider ids that refused a call this trial
        self.arrival = 0        # client requests seen: the manifest is in arrival order
        self.lock = threading.Lock()

    def _candidates(self, requires, json_wire_lock):
        found = []
        for order, provider in enumerate(self.brief.providers):
            probe = self.probes.get(provider.id)
            if probe is None:
                continue
            if json_wire_lock and provider.wire == "anthropic":
                continue
            for model in probe.models:
                if not model_free(self.brief, provider.id, model):
                    continue
                context, tools, vision, json_ok = model_caps(self.brief, provider.id, model)
                if requires.get("tools") and tools is not True:
                    continue
                if requires.get("vision") and vision is not True:
                    continue
                if requires.get("json") and json_ok is not True:
                    continue
                if requires.get("min_context") is not None and \
                        (context is None or context < requires["min_context"]):
                    continue
                row = self.brief.ratings.get((provider.id, model.id))
                rating = row["rating_milli"] if row is not None else 0
                found.append({
                    "provider": provider, "model": model.id, "rating": rating,
                    "order": order, "deferred": provider.id in self.deferred,
                })
        # Best rating first. A refusal demotes its provider only among equals:
        # a scheduled one-call refusal must not cost a later request the best
        # model, while a spent or out provider still loses every tie.
        found.sort(key=lambda c: (-c["rating"], c["deferred"], c["order"], c["model"]))
        return found

    def serve(self, client_api, body):
        """Answer one client request: (answer, model id), or (None, None)."""
        if client_api == "openai":
            neutral = neutral_from_openai(body)
        else:
            neutral = neutral_from_anthropic(body)
        with self.lock:
            index = self.arrival
            self.arrival += 1
        requires = {}
        if index < len(self.brief.manifest):
            requires.update(self.brief.manifest[index])
        for key, value in requirements_from_neutral(neutral).items():
            if value:
                requires[key] = True
        candidates = self._candidates(requires, neutral.json_mode)
        attempts = 0
        for candidate in candidates:
            if attempts >= MAX_ATTEMPTS:
                break
            attempts += 1
            provider = candidate["provider"]
            status, parsed, _ = http_call(
                "POST", upstream_url(provider, candidate["model"]),
                headers=upstream_headers(provider),
                body=upstream_body(provider.wire, candidate["model"], neutral))
            if status < 200 or status >= 300:
                if status in (402, 429) or status >= 500:
                    self.deferred.add(provider.id)
                continue
            answer = parse_answer(provider.wire, parsed)
            if answer is None:
                self.deferred.add(provider.id)
                continue
            return answer, candidate["model"]
        return None, None

    def served_model_ids(self):
        ids = []
        seen = set()
        for entry in self.catalog["providers"]:
            for model in entry["models"]:
                if model["free"] and model["id"] not in seen:
                    seen.add(model["id"])
                    ids.append((model["id"], entry["id"]))
        return ids

    def models_payload(self, anthropic):
        listed = self.served_model_ids()
        display = {}
        for probe in self.probes.values():
            for model in probe.models:
                display.setdefault(model.id, model.display_name)
        if anthropic:
            return {"data": [{"type": "model", "id": model_id,
                              "display_name": display.get(model_id, model_id),
                              "created_at": FIXED_CREATED_AT}
                             for model_id, _ in listed],
                    "has_more": False}
        return {"object": "list",
                "data": [{"id": model_id, "object": "model", "created": FIXED_CREATED,
                          "owned_by": provider_id} for model_id, provider_id in listed]}


# ---------------------------------------------------------------------------
# Client answers: JSON, or SSE synthesized from the whole provider answer.
# ---------------------------------------------------------------------------

def _pieces(text):
    if not text:
        return []
    return [text[i:i + SSE_PIECE_CHARS] for i in range(0, len(text), SSE_PIECE_CHARS)]


def _sse(data, event=None):
    return "%sdata: %s\n\n" % (("event: %s\n" % event) if event else "",
                               json.dumps(data, separators=(",", ":")))


def openai_json(answer, model_id):
    return {"id": "chatcmpl-agg", "object": "chat.completion", "created": FIXED_CREATED,
            "model": model_id,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": answer["text"]},
                         "finish_reason": _TO_OPENAI_STOP.get(answer["stop"], "stop"),
                         "logprobs": None}],
            "usage": {"prompt_tokens": answer["input"],
                      "completion_tokens": answer["output"],
                      "total_tokens": answer["input"] + answer["output"]}}


def openai_sse(answer, model_id):
    stop = _TO_OPENAI_STOP.get(answer["stop"], "stop")

    def chunk(delta, finish):
        return {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                "created": FIXED_CREATED, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = [_sse(chunk({"role": "assistant", "content": ""}, None))]
    for piece in _pieces(answer["text"]):
        frames.append(_sse(chunk({"content": piece}, None)))
    frames.append(_sse(chunk({}, stop)))
    frames.append(_sse({"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                        "created": FIXED_CREATED, "model": model_id, "choices": [],
                        "usage": {"prompt_tokens": answer["input"],
                                  "completion_tokens": answer["output"],
                                  "total_tokens": answer["input"] + answer["output"]}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json(answer, model_id):
    return {"id": "msg_agg", "type": "message", "role": "assistant", "model": model_id,
            "content": [{"type": "text", "text": answer["text"]}],
            "stop_reason": _TO_ANTHROPIC_STOP.get(answer["stop"], "end_turn"),
            "stop_sequence": None,
            "usage": {"input_tokens": answer["input"], "output_tokens": answer["output"]}}


def anthropic_sse(answer, model_id):
    stop = _TO_ANTHROPIC_STOP.get(answer["stop"], "end_turn")
    frames = [
        _sse({"type": "message_start",
              "message": {"id": "msg_agg", "type": "message", "role": "assistant",
                          "model": model_id, "content": [], "stop_reason": None,
                          "stop_sequence": None,
                          "usage": {"input_tokens": answer["input"], "output_tokens": 0}}},
             event="message_start"),
        _sse({"type": "content_block_start", "index": 0,
              "content_block": {"type": "text", "text": ""}}, event="content_block_start"),
    ]
    for piece in _pieces(answer["text"]):
        frames.append(_sse({"type": "content_block_delta", "index": 0,
                            "delta": {"type": "text_delta", "text": piece}},
                           event="content_block_delta"))
    frames.append(_sse({"type": "content_block_stop", "index": 0},
                       event="content_block_stop"))
    frames.append(_sse({"type": "message_delta",
                        "delta": {"stop_reason": stop, "stop_sequence": None},
                        "usage": {"output_tokens": answer["output"]}},
                       event="message_delta"))
    frames.append(_sse({"type": "message_stop"}, event="message_stop"))
    return "".join(frames)


def openai_error(status, message, error_type="api_error", code=None):
    return status, {"error": {"message": message, "type": error_type, "code": code}}


def anthropic_error(status, message, error_type="api_error"):
    return status, {"type": "error", "error": {"type": error_type, "message": message}}


# ---------------------------------------------------------------------------
# The HTTP server and the trial lifecycle.
# ---------------------------------------------------------------------------

AGGREGATOR = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        if isinstance(payload, (dict, list)):
            data = json.dumps(payload).encode("utf-8")
        else:
            data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, payload):
        self._send(status, "application/json", payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            anthropic = self.headers.get("x-api-key") is not None
            self._send_json(200, AGGREGATOR.models_payload(anthropic))
            return
        self._send_json(404, {"error": {"message": "no route %s" % path,
                                        "type": "invalid_request_error", "code": None}})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_json(404, {"error": {"message": "no route %s" % path,
                                            "type": "invalid_request_error", "code": None}})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if 0 < length <= MAX_BODY_BYTES else b""
        try:
            body = json.loads(raw.decode("utf-8")) if raw else None
        except ValueError:
            body = None
        if not is_record(body):
            if api == "openai":
                status, payload = openai_error(
                    400, "the request body is not one JSON object", "invalid_request_error")
            else:
                status, payload = anthropic_error(
                    400, "the request body is not one JSON object", "invalid_request_error")
            self._send_json(status, payload)
            return
        answer, model_id = AGGREGATOR.serve(api, body)
        if answer is None:
            if api == "openai":
                status, payload = openai_error(
                    503, "no free model on the directory could serve this request",
                    "api_error", "no_free_model")
            else:
                status, payload = anthropic_error(
                    503, "no free model on the directory could serve this request",
                    "overloaded_error")
            self._send_json(status, payload)
            return
        stream = body.get("stream") is True
        if api == "openai":
            if stream:
                self._send(200, "text/event-stream", openai_sse(answer, model_id))
            else:
                self._send_json(200, openai_json(answer, model_id))
        else:
            if stream:
                self._send(200, "text/event-stream", anthropic_sse(answer, model_id))
            else:
                self._send_json(200, anthropic_json(answer, model_id))


def register(port, catalog):
    payload = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    status, body, _ = http_call("POST", ENV_URL + "/register", body=payload)
    if status < 200 or status >= 300 or not is_record(body) or body.get("ok") is not True:
        raise RuntimeError("register answered HTTP %s" % status)
    return body


def main(argv):
    global AGGREGATOR
    brief = Brief(PROMPT_FILE)
    if brief.error:
        log(brief.error)
    probes = []
    catalog = {"providers": []}
    if brief.providers and ENV_URL:
        probes = [probe_provider(p) for p in brief.providers]
        catalog = build_catalog(brief, probes)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        return 0
    if not ENV_URL:
        sys.stdout.write(json.dumps({"providers": []}, separators=(",", ":")) + "\n")
        return 0
    AGGREGATOR = Aggregator(brief, probes, catalog)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout")
        return 1
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        reply = register(port, catalog)
    except RuntimeError as err:
        log(str(err))
        server.shutdown()
        return 1
    log("listening on 0.0.0.0:%d; registered for %s requests" % (port, reply.get("requests")))
    while True:
        status, body, _ = http_call("GET", ENV_URL + "/work", timeout=PROBE_TIMEOUT_S)
        if status == 200 and is_record(body) and body.get("done") is True:
            break
        time.sleep(POLL_INTERVAL_S)
    server.shutdown()
    log("the workload is done; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
