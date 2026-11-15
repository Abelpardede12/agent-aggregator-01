[![free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)

# The Best Free AI Model Aggregator

One server that speaks both client APIs — OpenAI Chat Completions and
Anthropic Messages, JSON or streaming — and routes every request to the best
free model across ten free-tier providers, translating between the OpenAI,
Anthropic and Gemini wires as needed.

## How it works

1. **Reads the brief.** The trial brief at `$MAKEFUN_PROMPT_FILE` carries a
   provider directory (wire, auth scheme, catalog path), a quality table
   (rating, context length, tool/vision/JSON support for every free model)
   and a workload manifest (the requirements of each of the 20 requests).
2. **Builds the catalog by probing.** Every provider's own catalog route is
   read back — catalog reads are free and spend no budget — so the catalog
   registered with the env lists exactly the models each provider serves,
   in the provider's own order, free flags included.
3. **Registers and serves.** The server binds an ephemeral port, registers
   `{protocol_version: 1, port, catalog}` with `$MAKEFUN_ENV_URL`, and
   answers `POST /v1/chat/completions`, `POST /v1/messages` and
   `GET /v1/models` (OpenAI shape for a Bearer header, Anthropic shape for
   an `x-api-key` header).
4. **Routes by quality, fails over on refusal.** For each request the free
   models that meet its requirements (tools, vision, JSON mode, minimum
   context — taken from the manifest row for that request, unioned with
   what the body itself shows) are ranked by rating. The top model is
   called; on a 402/429/5xx the request fails over to the next best. A
   refusal is treated as a provider-level signal, so the refusing
   provider's other models are deferred behind the rest. Providers are
   always called non-streaming; streamed answers are rendered from the
   complete provider answer in the client's own event shapes.
5. **Preserves what the provider saw.** Requests are translated through a
   neutral form that keeps the system text, the last user message, tool
   names and order, `max_tokens`, stop sequences, image bytes and JSON
   mode intact, and answers carry the provider's exact text, token usage
   and stop reason (mapped across the stop/end_turn/STOP families). For a
   JSON-mode OpenAI request the candidates are restricted to the OpenAI
   and Gemini wires, whose echoes can carry `json_mode` back.
6. **Exits clean.** The server polls `GET /work` and exits 0 as soon as
   the env reports the workload done. No clock, randomness or state
   outside the brief influences an answer.

## Run it

Inside a trial (Python 3.13, standard library only — nothing to install):

```sh
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief> python3 main.py
```

Print the catalog document — the same document the server registers —
and exit:

```sh
MAKEFUN_ENV_URL=<env url> MAKEFUN_PROMPT_FILE=<brief> python3 main.py --list
```

Outside a trial, `--list` prints `{"providers":[]}`.

## Practice results

Against the kit's three practice trials (hidden budgets, failure
schedules and outages included): **20/20 requests valid and a score of 99
on each** — quality 99, served 100, wire 100, failover 100, catalog 100.

## License

MIT — see [LICENSE](LICENSE).
