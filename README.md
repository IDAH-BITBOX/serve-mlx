# mlx-moe-stream

**Run large MLX Mixture-of-Experts (MoE) models locally on an Apple-silicon
Mac, with routed expert weights streamed from SSD instead of keeping every
expert in Unified Memory.**

`mlx-moe-stream` is a local, OpenAI-compatible server for supported MLX MoE
checkpoints. It preserves the model's router and expert computation: only the
experts selected for the current layer are read and materialized. The model's
non-expert shell (router, attention, shared/dense weights, and—in vision
mode—the vision tower) remains in Unified Memory.

한국어 사용 가이드: [README.ko.md](README.ko.md)

> This is pre-release software. It is designed for local, single-user serving
> on `127.0.0.1`, and it runs one generation at a time to keep memory use
> predictable.

## What you can do

| Need | Supported interface |
| --- | --- |
| Text completion and chat | `/v1/completions`, `/v1/chat/completions` |
| Token streaming | OpenAI-style SSE (`stream: true`) |
| Reasoning / thinking output | `reasoning_content` in chat responses when the model emits it |
| Function calling | OpenAI-style `tools` and `tool_choice`, for compatible tokenizer templates |
| Structured output | `response_format: json_object` or `json_schema` (non-streaming) |
| Image chat | Qwen3.6 and Gemma 4 with `--vision` and the optional VLM install |
| Several local models | Repeat `--model MODEL_ID=MANIFEST`; one model is resident at a time |

## Requirements

- Apple Silicon Mac and macOS
- Python 3.10 or newer
- Disk space for the MLX checkpoint and SSD reads during inference
- A supported MLX safetensors MoE checkpoint. The primary example in this guide
  is `mlx-community/Qwen3.6-35B-A3B-8bit`; the tested examples are:
  - `mlx-community/Qwen3.6-35B-A3B-8bit`
  - `mlx-community/Qwen3-30B-A3B-4bit`
  - `mlx-community/gemma-4-26b-a4b-it-8bit`

Large models can reside on SSD, but they are not zero-memory models: the
non-expert shell, the active routed experts, and the KV cache still use Unified
Memory. Start with the automatic settings and watch `/metrics`.

## M4 MacBook Pro (24GB) validation

End-to-end local serving and inference have been exercised on an Apple M4
MacBook Pro with 24GB Unified Memory using:

- `mlx-community/Qwen3.6-35B-A3B-8bit`
- `mlx-community/gemma-4-26b-a4b-it-8bit` (the exact Gemma 4 8-bit repository ID)

These are large MoE checkpoints: SSD streaming keeps inactive expert weights
out of Unified Memory, but the model shell, active experts, and KV cache still
need memory. Use the long-context configuration below when memory headroom is
tight; it trades prefill speed for a lower peak allocation.

### 256K context-window run

On that M4 system, Qwen3.6 completed a real OpenAI-compatible
`/v1/chat/completions` request with **262,143 prompt tokens and one generated
token** (**262,144 total tokens**). The run used `--resident-budget off`,
`--kv-cache 4bit`, and `--prefill-step-size 1024`; it took 2 h 3 m 3 s and
reported a 12.86 GiB MLX peak allocation. This verifies a 256K-class context
window, not an interactive 256K-token output. Full-window SSD-streamed MoE
prefills are intentionally a slow stress test.

### Sustained-workload and hardware-care notice

Long generations can keep the CPU/GPU and SSD I/O busy for hours. This is a
supported workload, not an instruction to defeat the Mac's thermal safeguards:
Apple laptops monitor internal temperature and cool critical components
automatically. It can make the enclosure warm and, on fan-equipped models,
raise fan speed. Use a stable, hard surface with unobstructed ventilation in a
10–35 °C room; never run it on bedding, a pillow, or inside a bag. See
[Apple's temperature guidance](https://support.apple.com/en-us/102336).

The main long-term consideration is the battery, not a single inference run.
Lithium-ion battery aging is affected by temperature history and charging
patterns. For repeated multi-hour serving, use a reliable power adapter, keep
**Optimized Battery Charging** / a sensible **Charge Limit** enabled, and pause
the job if the Mac becomes unusually hot or reports a temperature/charging
warning. See [Apple's Apple-silicon battery-health guidance](https://support.apple.com/en-mide/102589)
and [charge-limit guidance](https://support.apple.com/en-au/102338).

SSD-streamed inference is predominantly **read** I/O, so it is not normally a
material SSD-endurance concern. The writes to watch are model downloads,
repeated conversion/quantization, large traces, and application logs. Keep
those outputs bounded, and ensure free disk space remains available.

## Install

Until the project is published to PyPI, install the current public release
directly from GitHub. Do this in a Python 3.10+ virtual environment—do not
install it into the Homebrew/system Python:

```bash
python3.12 -m venv ~/.venvs/mlx-moe-stream
source ~/.venvs/mlx-moe-stream/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade \
  "mlx-moe-stream[vlm] @ git+https://github.com/IDAH-BITBOX/serve-mlx.git@main"
```

`[vlm]` installs image-chat support. For text-only serving, run this after
activating the virtual environment instead:

```bash
python3 -m pip install --upgrade \
  "mlx-moe-stream @ git+https://github.com/IDAH-BITBOX/serve-mlx.git@main"
```

To develop from a local checkout, clone the repository and install it in a
virtual environment instead:

```bash
git clone https://github.com/IDAH-BITBOX/serve-mlx.git
cd serve-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For image input in a local checkout, install the optional VLM dependency as
well:

```bash
python -m pip install -e ".[vlm]"
```

Verify the command is available:

```bash
mlx-moe-stream --help
```

## Quick start: text model

### 1. Prepare a manifest

The manifest indexes the exact safetensors byte ranges for each expert. It does
not copy model weights. Run it once for each model checkpoint:

```bash
mlx-moe-stream prepare \
  --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --output prepared-qwen3.6-35b
```

This creates `prepared-qwen3.6-35b/manifest.json`. Keep the manifest next to the
checkpoint cache or in your project; it is small and is safe to recreate with
the same checkpoint.

### 2. Start the local server

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --resident-budget auto \
  --kv-cache auto
```

The server keeps running until you press `Ctrl-C`; completing a request does
not unload the model or stop the server. The first request loads the model
shell and can take noticeably longer than later requests.

For MLX native-runtime stability, local requests run serially on one
long-lived server thread. A health or metrics request sent during generation
waits for that generation to finish; use a separate process if you need
independent monitoring.

### 3. Send a chat request

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-local",
    "messages": [{"role": "user", "content": "Explain sparse MoE routing in two sentences."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

Check health, registered models, and memory/KV decisions:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics
```

## Use it from the OpenAI Python SDK

Point an OpenAI-compatible client to the local base URL. No API key is needed,
but the SDK normally requires a non-empty placeholder value.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "Give me three uses of MoE models."}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### Stream tokens

```python
stream = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "Write a short haiku about SSDs."}],
    max_tokens=128,
    stream=True,
    stream_options={"include_usage": True},
)
for event in stream:
    text = event.choices[0].delta.content if event.choices else None
    if text:
        print(text, end="", flush=True)
```

## Image chat (Qwen3.6 or Gemma 4)

Prepare the model, install `[vlm]`, and start the server with `--vision`:

```bash
mlx-moe-stream prepare \
  --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --output prepared-qwen3.6-35b

mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --vision \
  --kv-cache auto
```

With `--vision`, the default profile deliberately disables the resident-expert
cache, reserves at least 2 GiB for transient VLM work, and uses a 256-token
prefill step. This leaves room for the vision tower and image-prefill
activations. You can override those values, but do so only after checking
`/metrics` on the target Mac.

### 16GB Mac: safe VLM starting point

For Qwen3.6-35B-A3B vision requests on a 16GB Mac, begin with a modest context
window and no resident expert cache:

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --vision \
  --resident-budget off \
  --memory-safety-margin auto \
  --scratch-reserve 2GiB \
  --kv-cache 4bit \
  --max-prompt-tokens 16384 \
  --max-tokens 128 \
  --prefill-step-size 256
```

The documented 256K text-context stress test ran on a 24GB Mac before adding
vision inputs. A 16GB Mac cannot be promised a 256K VLM context: the vision
tower, image features, KV cache, and long-prefill activations must fit in
Unified Memory together. If MLX reports insufficient memory, the API now
returns `503` with `code: "memory_pressure"` rather than an opaque `500`;
lower `--max-prompt-tokens` first, then the prefill step size.

Send a **direct image URL** (a URL that returns `image/jpeg`, `image/png`, and
so on), a `data:` URL, or a local file path in a user message. At most four
images are accepted per request. A Google share link or an HTML viewer page is
not an image URL; use the original/download image URL instead. Audio and video
are intentionally not supported.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-local",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
        {"type": "text", "text": "Describe this image briefly."}
      ]
    }],
    "max_tokens": 128
  }'
```

For Gemma 4, substitute this model ID when preparing:

```text
mlx-community/gemma-4-26b-a4b-it-8bit
```

## KV-cache precision and memory

The KV cache grows with context length. Choose its precision with
`--kv-cache` on both `serve` and `generate`:

```bash
--kv-cache auto   # default: choose the safest useful mode
--kv-cache bf16   # native, unquantized MLX KV cache
--kv-cache 8bit   # quantized KV cache; lower memory use
--kv-cache 4bit   # most compact KV cache; use for oversized context/model pressure
```

`auto` decides only after the real non-expert model shell has been loaded and
measured. Its policy is:

1. Enough post-shell Unified Memory headroom: use native BF16/FP KV cache.
2. BF16 does not fit the KV allowance: use 8-bit KV cache.
3. 8-bit does not fit (for example, an oversized model or maximum context): use 4-bit KV cache.

The resolved mode, estimated KV bytes, and reservation appear in server
`/metrics` and in the `generate` log / optional memory summary. The automatic
choice also reserves that estimate before allocating the expert cache, so
`--resident-budget auto` has less memory available when a larger KV cache is
needed.

For the `generate` command, tell the planner the largest total context you
expect (prompt plus completion):

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --prompt "Summarize this document..." \
  --max-tokens 512 \
  --kv-cache auto \
  --kv-max-context 8192 \
  --resident-budget auto
```

For `serve`, the reservation is calculated from
`--max-prompt-tokens + --max-tokens`; set those limits honestly. `--kv-reserve`
is a minimum safety floor, not the requested cache precision.

For a large context that otherwise reaches a Metal out-of-memory error during
prefill, lower `--prefill-step-size`. It controls how many prompt tokens are
processed at once, independently of the context limit. Smaller values reduce
temporary MoE activation memory but take longer:

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --resident-budget off \
  --kv-cache 4bit \
  --max-prompt-tokens 262143 \
  --max-tokens 1 \
  --prefill-step-size 1024
```

This is a **context-window** setting, not a promise of an interactive 256K-token
output. A full-window prefill can take a long time on SSD-streamed MoE models.

Quantized KV cache reduces memory use but can change generation quality. Use
`bf16` for quality-sensitive short contexts, and prefer `auto` or `8bit`/`4bit`
when serving a large model on a smaller-memory Mac.

### Preserve Unified Memory headroom

`--memory-safety-margin auto` is now the default. It reserves 25% of physical
Unified Memory from the expert cache: 2 GiB on an 8 GiB Mac, 4 GiB on a 16 GiB
Mac, and 6 GiB on a 24 GiB Mac (capped at 8 GiB). See the active decision in
`/metrics` under `memory_budget`.

For even more free memory, use an explicit cache cap and larger reserves:

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --vision \
  --resident-budget 2GiB \
  --memory-safety-margin 8GiB \
  --scratch-reserve 2GiB \
  --kv-cache 4bit \
  --max-prompt-tokens 2048 \
  --max-tokens 128
```

The safety margin is deliberately excluded from the resident expert cache; it
is not a promise that every model can fit. In particular, a 35B model may have
a non-expert shell too large for an 8 GiB or 16 GiB Mac. In that case M7 rejects
the expert-cache plan instead of using swap and making the machine unresponsive.

For the smallest sustained memory footprint, disable the resident expert cache
entirely with `--resident-budget off`. Experts still stream from SSD and are
materialized for the active layer, but are not retained afterward:

```bash
mlx-moe-stream serve --manifest prepared-qwen3.6-35b/manifest.json \
  --resident-budget off --kv-cache 4bit --max-prompt-tokens 1024 --max-tokens 64
```

## Multiple models

Register several prepared manifests with IDs. This exposes all of them through
`/v1/models`, but keeps only the currently requested engine in Unified Memory.
Switching models unloads the prior engine and loads the selected one.

```bash
mlx-moe-stream serve \
  --model qwen=prepared-qwen3.6-35b/manifest.json \
  --model gemma=prepared-gemma4/manifest.json \
  --model-id qwen \
  --vision \
  --resident-budget auto \
  --kv-cache auto
```

Use `"model": "qwen"` or `"model": "gemma"` in each API request. All
registered models share the server's memory, context, KV-cache, and `--vision`
settings; run separate server processes when models need different policies.

## Thinking, tools, and JSON output

For chat completions, the server passes tool definitions and thinking controls
to tokenizer templates that declare support for them. It converts compatible
model output into OpenAI-style `reasoning_content` and `tool_calls` fields.

```python
response = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "What is the weather in Seoul?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up current weather.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    tool_choice="auto",
    reasoning_effort="medium",
)
print(response.choices[0].message.tool_calls)
```

`response_format` supports `json_object` and `json_schema` for non-streaming
chat requests. JSON is validated before the response is returned; streaming
structured output is deliberately rejected because it cannot be validated
safely before the final token.

## One-shot generation and routing tools

Use `generate` for a direct text prompt without starting HTTP serving:

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 64 \
  --resident-budget auto \
  --kv-cache auto
```

Use `trace` to inspect Qwen3-MoE router choices and `simulate` to examine
expert-cache locality. `trace` currently targets the Qwen3-MoE family, so this
separate example uses Qwen3-30B:

```bash
mlx-moe-stream trace \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 64 \
  --output routes.jsonl \
  --summary routes-summary.json

mlx-moe-stream simulate --trace routes.jsonl
```

## Practical tuning order

1. Start with `--resident-budget auto --kv-cache auto`.
2. Keep `--max-prompt-tokens` and `--max-tokens` near real usage.
3. If memory pressure or swapping appears, reduce context first, then select
   `--kv-cache 8bit` or `--kv-cache 4bit`.
4. If responses are slow but memory is healthy, increase SSD locality with a
   reasonable resident expert budget; do not allocate beyond the reported
   safe budget.
5. Keep the server on loopback unless you add your own authenticated reverse
   proxy. This server intentionally has no authentication.

## Development

Install test tools with `python -m pip install -e ".[dev,vlm]"`, then run:

```bash
ruff check src tests
pytest
```

The project is licensed under Apache-2.0.
