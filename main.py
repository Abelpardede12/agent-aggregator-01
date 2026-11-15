#!/usr/bin/env python3
"""Free-model aggregator: one server, OpenAI + Anthropic wires, over the
trial's ten simulated free-tier providers (OpenAI / Anthropic / Gemini wires).

Reads the brief at $MAKEFUN_PROMPT_FILE (providers, sealed quality table,
workload manifest), probes each provider's own catalog route, registers the
catalog at $MAKEFUN_ENV_URL, routes each request to the highest-rated free
model meeting its requirements, fails over in rating order, and exits 0 when
the env says the workload is done. Python standard library only.
"""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CREATED = 1767225600  # fixed timestamp; no wall clock ever reaches a request


# ---------------------------------------------------------------- brief

def extract_docs(text):
    """Pull the three embedded JSON documents out of the brief text."""
    docs = {}
    i = 0
    n = len(text)
    while True:
        j = text.find('{"schema":"free-model-aggregator-01/', i)
        if j < 0:
            break
        depth = 0
        k = j
        in_str = False
        esc = False
        while k < n:
            c = text[k]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
            k += 1
        try:
            doc = json.loads(text[j:k + 1])
        except Exception:
            i = j + 1
            continue
        docs[doc["schema"].split("/")[-2]] = doc
        i = k + 1
    return docs


def load_brief():
    path = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    docs = extract_docs(text)
    providers = docs.get("providers", {}).get("providers", [])
    ratings = docs.get("ratings", {}).get("ratings", [])
    manifest = docs.get("workload", {}).get("manifest", [])
    return providers, ratings, manifest


# ---------------------------------------------------------------- http util

def http_call(method, url, headers=None, body=None, timeout=15):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        raise IOError(str(e))


def auth_headers(provider):
    auth = provider.get("auth")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


# ---------------------------------------------------------------- catalog probe

def probe_catalog(env_url, provider):
    """Ask the provider's own catalog route, in its own wire."""
    base = env_url + provider["path"]
    wire = provider["wire"]
    if wire == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    status, raw = http_call("GET", url, headers=auth_headers(provider))
    if status != 200:
        raise IOError("catalog %s -> %s" % (provider["id"], status))
    doc = json.loads(raw.decode("utf-8"))
    models = []
    if wire == "gemini":
        items = doc.get("models", [])
        for m in items:
            name = m.get("name", "")
            mid = name[len("models/"):] if name.startswith("models/") else name
            models.append({"id": mid, "free": bool(m.get("free")),
                           "context_length": m.get("context_length")})
    else:
        for m in doc.get("data", []):
            models.append({"id": m.get("id"), "free": bool(m.get("free")),
                           "context_length": m.get("context_length")})
    return models


def build_catalog(env_url, providers):
    out = {"providers": []}
    for p in providers:
        models = probe_catalog(env_url, p)
        out["providers"].append({
            "id": p["id"],
            "base_url": p["path"],
            "key_env": p.get("key_env", ""),
            "wire": p["wire"],
            "models": models,
        })
    return out


# ---------------------------------------------------------------- neutral form

def text_of(content):
    """Join the text blocks of an OpenAI/Anthropic content value."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("type") == "text"
                     and isinstance(b.get("text"), str))


def neutral_from_openai(body):
    system_parts = []
    messages = []
    for m in body.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            t = text_of(content)
            if t:
                system_parts.append(t)
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image_url" and isinstance(b.get("image_url"), dict):
                    parts.append({"type": "image_url",
                                  "url": b["image_url"].get("url", "")})
        messages.append({"role": "assistant" if role == "assistant" else "user",
                         "parts": parts})
    tools = []
    for t in body.get("tools", []) or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            tools.append({"name": fn["name"],
                          "description": fn.get("description", ""),
                          "parameters": fn.get("parameters", {})})
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (
        [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else [])
    fmt = body.get("response_format") if isinstance(body.get("response_format"), dict) else {}
    mt = body.get("max_tokens")
    if mt is None:
        mt = body.get("max_completion_tokens")
    return {"system": "\n".join(system_parts), "messages": messages,
            "tools": tools, "max_tokens": mt if isinstance(mt, (int, float)) else None,
            "stops": stops, "json": fmt.get("type") in ("json_object", "json_schema"),
            "temperature": body.get("temperature"),
            "stream": body.get("stream") is True}


def neutral_from_anthropic(body):
    system = text_of(body.get("system"))
    messages = []
    for m in body.get("messages", []) or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image" and isinstance(b.get("source"), dict):
                    src = b["source"]
                    if src.get("type") == "base64" and isinstance(src.get("data"), str):
                        parts.append({"type": "image", "data": src["data"],
                                      "media": src.get("media_type", "image/png")})
                    elif src.get("type") == "url" and isinstance(src.get("url"), str):
                        parts.append({"type": "image_url", "url": src["url"]})
        messages.append({"role": "assistant" if role == "assistant" else "user",
                         "parts": parts})
    tools = []
    for t in body.get("tools", []) or []:
        if isinstance(t, dict) and isinstance(t.get("name"), str):
            tools.append({"name": t["name"],
                          "description": t.get("description", ""),
                          "parameters": t.get("input_schema", {})})
    stops = body.get("stop_sequences")
    stops = [s for s in stops if isinstance(s, str)] if isinstance(stops, list) else []
    mt = body.get("max_tokens")
    return {"system": system, "messages": messages, "tools": tools,
            "max_tokens": mt if isinstance(mt, (int, float)) else None,
            "stops": stops, "json": False,
            "temperature": body.get("temperature"),
            "stream": body.get("stream") is True}


def body_requirements(neutral):
    req = {}
    if neutral["tools"]:
        req["tools"] = True
    for m in neutral["messages"]:
        for p in m["parts"]:
            if p["type"] in ("image", "image_url"):
                req["vision"] = True
                break
    if neutral["json"]:
        req["json"] = True
    return req


# ---------------------------------------------------------------- wire builders

def openai_messages(neutral):
    out = []
    if neutral["system"]:
        out.append({"role": "system", "content": neutral["system"]})
    for m in neutral["messages"]:
        parts = m["parts"]
        if len(parts) == 1 and parts[0]["type"] == "text":
            out.append({"role": m["role"], "content": parts[0]["text"]})
            continue
        content = []
        for p in parts:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image_url":
                content.append({"type": "image_url", "image_url": {"url": p["url"]}})
            elif p["type"] == "image":
                content.append({"type": "image_url", "image_url": {
                    "url": "data:%s;base64,%s" % (p["media"], p["data"])}})
        out.append({"role": m["role"], "content": content})
    return out


def call_openai(env_url, provider, model, neutral):
    body = {"model": model, "messages": openai_messages(neutral),
            "stream": False}
    if neutral["max_tokens"] is not None:
        body["max_tokens"] = neutral["max_tokens"]
    if neutral["temperature"] is not None:
        body["temperature"] = neutral["temperature"]
    if neutral["tools"]:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}} for t in neutral["tools"]]
        body["tool_choice"] = "auto"
    if neutral["stops"]:
        body["stop"] = neutral["stops"]
    if neutral["json"]:
        body["response_format"] = {"type": "json_object"}
    status, raw = http_call("POST", env_url + provider["path"] + "/v1/chat/completions",
                            headers=auth_headers(provider), body=body)
    if status != 200:
        raise ProviderRefusal(status)
    doc = json.loads(raw.decode("utf-8"))
    ch = doc["choices"][0]
    msg = ch.get("message", {})
    content = msg.get("content")
    text = content if isinstance(content, str) else text_of(content)
    usage = doc.get("usage", {})
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens")


def call_anthropic(env_url, provider, model, neutral):
    msgs = []
    for m in neutral["messages"]:
        content = []
        for p in m["parts"]:
            if p["type"] == "text":
                content.append({"type": "text", "text": p["text"]})
            elif p["type"] == "image":
                content.append({"type": "image", "source": {
                    "type": "base64", "media_type": p["media"], "data": p["data"]}})
            elif p["type"] == "image_url":
                content.append({"type": "image", "source": {
                    "type": "url", "url": p["url"]}})
        msgs.append({"role": m["role"], "content": content})
    body = {"model": model, "messages": msgs,
            "max_tokens": neutral["max_tokens"] if neutral["max_tokens"] is not None else 1024,
            "stream": False}
    if neutral["system"]:
        body["system"] = neutral["system"]
    if neutral["temperature"] is not None:
        body["temperature"] = neutral["temperature"]
    if neutral["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"]} for t in neutral["tools"]]
    if neutral["stops"]:
        body["stop_sequences"] = neutral["stops"]
    status, raw = http_call("POST", env_url + provider["path"] + "/v1/messages",
                            headers={**auth_headers(provider),
                                     "anthropic-version": "2023-06-01"},
                            body=body)
    if status != 200:
        raise ProviderRefusal(status)
    doc = json.loads(raw.decode("utf-8"))
    usage = doc.get("usage", {})
    return text_of(doc.get("content")), usage.get("input_tokens"), usage.get("output_tokens")


def call_gemini(env_url, provider, model, neutral):
    contents = []
    for m in neutral["messages"]:
        parts = []
        for p in m["parts"]:
            if p["type"] == "text":
                parts.append({"text": p["text"]})
            elif p["type"] == "image":
                parts.append({"inlineData": {"mimeType": p["media"], "data": p["data"]}})
            elif p["type"] == "image_url":
                parts.append({"fileData": {"fileUri": p["url"]}})
        contents.append({"role": "model" if m["role"] == "assistant" else "user",
                         "parts": parts})
    config = {}
    if neutral["max_tokens"] is not None:
        config["maxOutputTokens"] = neutral["max_tokens"]
    if neutral["stops"]:
        config["stopSequences"] = neutral["stops"]
    if neutral["json"]:
        config["responseMimeType"] = "application/json"
    if neutral["temperature"] is not None:
        config["temperature"] = neutral["temperature"]
    body = {"contents": contents}
    if neutral["system"]:
        body["systemInstruction"] = {"parts": [{"text": neutral["system"]}]}
    if neutral["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"]} for t in neutral["tools"]]}]
    if config:
        body["generationConfig"] = config
    url = (env_url + provider["path"] + "/v1beta/models/"
           + urllib.parse.quote(model, safe="") + ":generateContent")
    status, raw = http_call("POST", url, headers=auth_headers(provider), body=body)
    if status != 200:
        raise ProviderRefusal(status)
    doc = json.loads(raw.decode("utf-8"))
    parts = doc["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = doc.get("usageMetadata", {})
    return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")


class ProviderRefusal(Exception):
    def __init__(self, status):
        super().__init__("provider refused %s" % status)
        self.status = status


CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "gemini": call_gemini}


# ---------------------------------------------------------------- routing

class Router:
    def __init__(self, env_url, providers, ratings, catalog):
        self.env_url = env_url
        self.by_id = {p["id"]: p for p in providers}
        self.catalog = catalog
        self.catalog_free = {}
        for p in catalog["providers"]:
            self.catalog_free[p["id"]] = {
                m["id"] for m in p["models"] if m.get("free") is True}
        self.rows = []
        for r in ratings:
            if not r.get("free"):
                continue
            free = self.catalog_free.get(r["provider"])
            if free is not None and r["model"] not in free:
                continue
            self.rows.append(r)
        self.refused = set()
        self._lock = threading.Lock()
        self._served = 0
        self._manifest = []

    def next_requirements(self):
        with self._lock:
            idx = self._served
            self._served += 1
        if 0 <= idx < len(self._manifest):
            return dict(self._manifest[idx])
        return {}

    def candidates(self, req):
        out = []
        for r in self.rows:
            if req.get("tools") and not r.get("tools"):
                continue
            if req.get("vision") and not r.get("vision"):
                continue
            if req.get("json") and not r.get("json"):
                continue
            if req.get("min_context") is not None and (
                    r.get("context_length") or 0) < req["min_context"]:
                continue
            out.append(r)
        # highest rating first; a refusal demotes its provider only among
        # equal ratings (never a permanent deferral)
        out.sort(key=lambda r: (-r["rating_milli"],
                                1 if r["provider"] in self.refused else 0))
        return out

    def serve(self, neutral, req):
        """Try candidates in order; return (text, in_tok, out_tok, model)."""
        for row in self.candidates(req):
            provider = self.by_id[row["provider"]]
            # a JSON-mode OpenAI request cannot ride an Anthropic wire:
            # that wire echoes json_mode false
            if neutral["json"] and provider["wire"] == "anthropic":
                continue
            try:
                text, i, o = CALLERS[provider["wire"]](
                    self.env_url, provider, row["model"], neutral)
            except ProviderRefusal as e:
                if e.status in (402, 429) or e.status >= 500:
                    self.refused.add(provider["id"])
                continue
            except Exception:
                self.refused.add(provider["id"])
                continue
            return text, i, o, row["model"]
        raise IOError("no provider served the request")


# ---------------------------------------------------------------- client wire

def split_text(text):
    return [text[i:i + 13] for i in range(0, len(text), 13)] or [""]


def openai_response(model, text, i, o, stream):
    usage = {"prompt_tokens": i, "completion_tokens": o,
             "total_tokens": (i or 0) + (o or 0)}
    if not stream:
        return 200, {"Content-Type": "application/json"}, json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion",
            "created": CREATED, "model": model,
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop", "logprobs": None}],
            "usage": usage}).encode("utf-8")
    frames = []
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": CREATED, "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                     "finish_reason": None}]}))
    for piece in split_text(text):
        frames.append("data: %s\n\n" % json.dumps({
            "id": "chatcmpl-sim", "object": "chat.completion.chunk",
            "created": CREATED, "model": model,
            "choices": [{"index": 0, "delta": {"content": piece},
                         "finish_reason": None}]}))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": CREATED, "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-sim", "object": "chat.completion.chunk",
        "created": CREATED, "model": model, "choices": [], "usage": usage}))
    frames.append("data: [DONE]\n\n")
    return 200, {"Content-Type": "text/event-stream; charset=utf-8",
                 "Cache-Control": "no-cache"}, "".join(frames).encode("utf-8")


def anthropic_response(model, text, i, o, stream):
    if not stream:
        return 200, {"Content-Type": "application/json"}, json.dumps({
            "id": "msg_sim", "type": "message", "role": "assistant",
            "model": model, "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": i, "output_tokens": o}}).encode("utf-8")
    frames = []

    def ev(event, data):
        frames.append("event: %s\ndata: %s\n\n" % (event, json.dumps(data)))
    ev("message_start", {"type": "message_start", "message": {
        "id": "msg_sim", "type": "message", "role": "assistant",
        "model": model, "content": [], "stop_reason": None,
        "stop_sequence": None, "usage": {"input_tokens": i, "output_tokens": 0}}})
    ev("content_block_start", {"type": "content_block_start", "index": 0,
                               "content_block": {"type": "text", "text": ""}})
    for piece in split_text(text):
        ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                   "delta": {"type": "text_delta", "text": piece}})
    ev("content_block_stop", {"type": "content_block_stop", "index": 0})
    ev("message_delta", {"type": "message_delta",
                         "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                         "usage": {"output_tokens": o}})
    ev("message_stop", {"type": "message_stop"})
    return 200, {"Content-Type": "text/event-stream; charset=utf-8",
                 "Cache-Control": "no-cache"}, "".join(frames).encode("utf-8")


# ---------------------------------------------------------------- server

class Handler(BaseHTTPRequestHandler):
    router = None      # set by main()
    manifest = {}      # request id -> requirements
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, headers, body):
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status, obj):
        self._send(status, {"Content-Type": "application/json"},
                   (json.dumps(obj) + "\n").encode("utf-8"))

    def do_GET(self):
        if self.path == "/v1/models" or self.path.startswith("/v1/models?"):
            xkey = self.headers.get("x-api-key")
            ids = []
            seen = set()
            for p in self.router.catalog["providers"]:
                for m in p["models"]:
                    if m.get("free") is True and m["id"] not in seen:
                        seen.add(m["id"])
                        ids.append(m["id"])
            if xkey:
                data = [{"type": "model", "id": mid, "display_name": mid,
                         "created_at": "2026-01-01T00:00:00Z", "free": True}
                        for mid in ids]
                self._json(200, {"data": data, "has_more": False})
            else:
                data = [{"id": mid, "object": "model", "created": CREATED,
                         "owned_by": "aggregator", "free": True} for mid in ids]
                self._json(200, {"object": "list", "data": data})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json(400, {"error": "bad_request"})
            return
        if self.path == "/v1/chat/completions":
            neutral = neutral_from_openai(body)
            api = "openai"
        elif self.path == "/v1/messages":
            neutral = neutral_from_anthropic(body)
            api = "anthropic"
        else:
            self._json(404, {"error": "not_found"})
            return
        # requirements: the driver sends one request at a time, in manifest
        # order, so this POST is the next manifest row; union the body's
        req = self.router.next_requirements()
        req.update(body_requirements(neutral))
        try:
            text, i, o, model = self.router.serve(neutral, req)
        except Exception as e:
            self._json(502, {"error": str(e)})
            return
        if api == "openai":
            status, headers, payload = openai_response(model, text, i, o,
                                                       neutral["stream"])
        else:
            status, headers, payload = anthropic_response(model, text, i, o,
                                                         neutral["stream"])
        self._send(status, headers, payload)


# ---------------------------------------------------------------- main

def main():
    brief = load_brief()
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if "--list" in sys.argv:
        if brief is None or not env_url:
            print(json.dumps({"providers": []}))
            return 0
        providers, _, _ = brief
        try:
            catalog = build_catalog(env_url, providers)
        except Exception:
            print(json.dumps({"providers": []}))
            return 0
        print(json.dumps(catalog))
        return 0
    if brief is None or not env_url:
        sys.stderr.write("no brief (MAKEFUN_PROMPT_FILE) or env (MAKEFUN_ENV_URL)\n")
        return 1
    providers, ratings, manifest = brief
    catalog = build_catalog(env_url, providers)
    router = Router(env_url, providers, ratings, catalog)
    router._manifest = [m.get("requires", {}) for m in manifest]
    Handler.router = router
    Handler.manifest = router._manifest
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # register (retry briefly: the env may still be starting)
    registered = False
    for _ in range(20):
        try:
            status, _ = http_call("POST", env_url + "/register",
                                  body={"protocol_version": 1, "port": port,
                                        "catalog": catalog}, timeout=5)
            if status == 200:
                registered = True
                break
        except Exception:
            pass
        time.sleep(0.5)
    if not registered:
        sys.stderr.write("register failed\n")
        server.shutdown()
        return 1
    # poll /work until the workload is done
    while True:
        try:
            status, raw = http_call("GET", env_url + "/work", timeout=5)
            if status == 200:
                doc = json.loads(raw.decode("utf-8"))
                if doc.get("done") is True:
                    break
        except Exception:
            pass
        time.sleep(0.25)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
