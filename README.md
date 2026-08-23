# mlx-moe-stream

`mlx-moe-stream` is an Apple-Silicon-first project for exact out-of-core MoE
inference on top of [mlx-lm](https://github.com/ml-explore/mlx-lm). It keeps
non-expert weights on the normal mlx-lm path and will stream only routed expert
weights into an explicitly budgeted resident working set.

The repository is at the M10 predictive-prefetch milestone. It can trace
Qwen3-MoE routing, create a validated manifest for exact expert-only disk
reads, execute the exact streaming MoE path, retain routed experts in an
explicit byte-budgeted LRU cache, and materialize each routed prefill expert
once per layer. It also overlaps known exact expert reads with MLX GPU work.
M7 measures the real quantized non-expert shell, derives a safe expert-cache
budget from MLX's recommended working set, and records memory pressure actions.
M8 exposes one loaded engine through a bounded localhost OpenAI-compatible API.
M8.5 dispatches a prepared checkpoint to an exact Qwen3, Qwen3.5, or Gemma 4
text adapter. M9 registers multiple prepared manifests but holds at most one
engine in Unified Memory. M10 adds strictly bounded, trace-trained next-layer
expert prefetch. It does **not** yet implement multimodal inputs.

## Requirements

- Apple Silicon and macOS
- Python 3.10+
- A supported MLX safetensors MoE checkpoint, locally available or from a
  Hugging Face repository

## Install

```bash
python3.12 -m pip install -e ".[dev]"
mlx-moe-stream --help
pytest
```

## Capture a routing trace

The tracer uses the normal mlx-lm execution path. It only observes router
top-k indices and scores, so enabling it does not change model routing or
expert execution.

```bash
mlx-moe-stream trace \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --prompt "Explain sparse mixture-of-experts routing." \
  --max-tokens 64 \
  --output routes.jsonl \
  --summary routes-summary.json
```

`routes.jsonl` has one event per routed token and MoE layer. Each event records
the request ID, prefill/decode phase, token and layer IDs, selected experts,
router scores, and a UTC timestamp.

The trace command uses greedy decoding and has an initial batch-size-one scope.
No remote Python model code is executed (`trust_remote_code=False`).

## Simulate cache locality

```bash
mlx-moe-stream simulate --trace routes.jsonl
```

The M1 simulator assumes equal-sized expert bundles. M2's manifest now records
the exact byte size of every expert bundle; byte-weighted cache simulation will
replace the baseline in M4.

## Prepare exact expert reads (M2 / M8.5)

```bash
mlx-moe-stream prepare \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --output prepared-qwen3

python benchmarks/storage_read.py \
  --manifest prepared-qwen3/manifest.json \
  --layer 0 --expert 0
```

`prepare` reads only safetensors headers and creates `manifest.json`; it does
not duplicate model shards or materialize weights in MLX. The storage benchmark
uses `os.pread()` for only the requested expert's tensor ranges and reports
the exact read bytes and count.

M8.5 recognizes these layouts automatically:

- `qwen3_moe` — Qwen3-MoE
- `qwen3_5_moe` — Qwen3.5-MoE text subtree, including its resident shared expert
- `gemma4` — Gemma 4 26B-A4B text subtree, using GeGLU routed experts

```bash
mlx-moe-stream prepare \
  --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --output prepared-qwen3.6-35b

mlx-moe-stream prepare \
  --model mlx-community/gemma-4-26b-a4b-it-8bit \
  --output prepared-gemma4-26b
```

These are multimodal checkpoints. M8.5 deliberately loads only their
`language_model` text weights; image inputs and vision-tower execution are not
exposed by `generate` or `serve`.

## Exact generation: M3–M5

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

Without `--resident-budget`, M3 keeps all routed experts on SSD and
materializes every selected expert for the active forward pass. This remains
the exact no-cache correctness baseline.

Set an explicit M4 cache budget to retain only selected expert arrays in the
Unified Memory working set. A hit never reads the SSD; a miss reads the same
exact manifest spans as M3. The cache never changes router choices or expert
math.

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --resident-budget 2GB \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

The budget is an expert-only cache limit. M7 validates explicit values against
the measured shell plus the configured KV, scratch, and OS safety reservations.
M4 still fails explicitly if even one required expert cannot fit or if every
evictable entry is pinned; it never silently falls back to approximate
inference.

## Expert-major prefill (M5)

For every prefill layer, M5 first collects all router choices, groups selected
tokens by expert, materializes that expert once, and restores outputs to their
original `(token, top-k)` slots. Decode remains token-major. `expert_major` is
the default whenever a model call has more than one input token; use
`token_major` as the exact M3/M4 prefill baseline.

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --resident-budget 2GB \
  --prefill-strategy expert_major \
  --prefill-order resident_first \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

The order can be `resident_first`, `expert_id`, or `disk_offset`. It affects
only load order, never router choices or output placement. MLX batched
quantized matmul does not exactly match repeated vector matmul for this model,
so M5 shares each expert's I/O/materialization but retains vector math per
selected token to preserve exact logits.

Measure the prefill baseline and M5 side-by-side. `--repeat` creates a longer
tokenized prompt while reporting its actual token count.

```bash
python benchmarks/prefill.py \
  --manifest prepared-qwen3/manifest.json \
  --prompt "Explain sparse MoE routing." \
  --repeat 32 \
  --resident-budget 2GB \
  --prefill-strategy expert_major \
  --prefill-order resident_first
```

## I/O and GPU overlap (M6)

M6 has a bounded priority I/O worker pool. Once the router has exposed an
exact current-layer expert set, it reads the next known expert in the
background while MLX executes the current expert. Duplicate demand/prefetch
requests coalesce to one `pread()` task. This is not predictive prefetch: a
miss still waits for and executes its actual routed expert, so output semantics
are unchanged.

M6 is explicit by default (`--io-workers 0` keeps the M5 sequential path):

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --resident-budget 2GB \
  --io-workers 1 \
  --prefetch-depth 1 \
  --async-gpu \
  --timeline m6-timeline.json \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

The timeline records `load_start`, `load_end`, `materialize_start`,
`materialize_end`, `gpu_enqueue`, and `gpu_done`. Compare M6's three required
paths on the same workload:

```bash
python benchmarks/overlap.py \
  --manifest prepared-qwen3/manifest.json \
  --prompt "Explain sparse MoE routing." \
  --io-workers 1 \
  --prefetch-depth 1
```

## Safe automatic Unified Memory budget (M7)

Use `--resident-budget auto` to measure the quantized non-expert shell after it
is loaded, then reserve capacity for the OS, the mlx-lm KV cache, and transient
MLX work before setting the expert-only LRU capacity:

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --resident-budget auto \
  --memory-safety-margin 2GB \
  --kv-reserve 1GB \
  --scratch-reserve 1GB \
  --memory-summary m7-memory.json \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

The automatic cache budget is:

```text
max_recommended_working_set
− safety_margin
− measured_nonexpert_shell
− KV_reserve
− scratch_reserve
```

`m7-memory.json` records the startup decision, MLX active/cache/peak memory,
process RSS, and macOS swap state before and after the request. Swap is only an
observability signal; it is never treated as cache capacity or a success path.
At a completed sparse-layer safe point, sustained working-set pressure first
turns off future prefetches, then evicts unpinned LRU experts, shrinks the
expert cache, and finally rejects the request with an explicit error. Router
choices and expert math are unchanged by all four actions.

`--wired-limit` calls `mx.set_wired_limit()` only when explicitly supplied. No
default mode changes macOS `sysctl` settings or requires administrator access.

## Local OpenAI-compatible server (M8 / M8.5 / M9)

M8 loads one prepared supported-MoE manifest once, enables the M7 automatic budget
by default, and serializes model execution to one active generation. It binds
only to `127.0.0.1` or `localhost`; unauthenticated non-loopback binds are
rejected deliberately.

```bash
mlx-moe-stream --verbose serve \
  --manifest prepared-qwen3/manifest.json \
  --model-id qwen3-local \
  --port 8000 \
  --max-prompt-tokens 4096 \
  --max-tokens 256
```

The API has these endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /metrics`

`/metrics` returns JSON aggregate counts plus the last request's TTFT,
prefill/decode tok/s, cache hit rate, disk bytes, resident expert bytes, and
MLX peak memory. A second simultaneous generation receives HTTP `429` rather
than sharing mutable KV or expert-cache state.

The M8 protocol supports non-streaming, greedy `n=1` requests. `stream=true`,
sampling controls, tools, and structured response formats are explicitly
rejected rather than silently ignored.

Use it with the OpenAI Python client (the client is optional; it is not a
server dependency):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local-unused")
reply = client.chat.completions.create(
    model="qwen3-local",
    messages=[{"role": "user", "content": "Explain sparse MoE routing."}],
    max_tokens=64,
)
print(reply.choices[0].message.content)
```

M8.5 can expose one prepared Qwen3-MoE, Qwen3.5-MoE, or Gemma 4 manifest at a
time. It is not a universal loader for dense models, image inputs, or arbitrary
MoE layouts. Adding a family still requires an adapter that proves selective
expert reads and preserves its specific router/expert semantics.

## Lazy model registry (M9)

Use repeated `--model MODEL_ID=MANIFEST` to make multiple prepared models
discoverable through `GET /v1/models`. The server starts with **no** loaded
engine. The first request for a model loads its text shell and streams only its
routed experts. A request for another model closes the prior engine before
opening the next one, so Qwen and Gemma shells never coexist in Unified Memory.

```bash
mlx-moe-stream --verbose serve \
  --model qwen3.6=prepared-qwen3.6-35b/manifest.json \
  --model gemma4=prepared-gemma4-26b/manifest.json \
  --model-id qwen3.6 \
  --port 8000
```

`--model-id` selects the default when a client omits `model`; each OpenAI
request may select any registered model ID explicitly. The same single active
generation limit applies during loading and inference. `/metrics` includes a
`registry` object with the active ID plus load, unload, switch, and load-failure
counters. The original single-model `--manifest ... --model-id ...` invocation
remains supported.

## Trace-trained predictive prefetch (M10)

M10 trains a conditional distribution from route traces: after the current
layer's router has selected experts, it may issue exact `pread()` requests for
likely experts in the **next** layer. It never changes router IDs, expert
weights, or output math. A wrong prediction is only a bounded unused I/O read.

Train a predictor from a trace, then attach it to generation or serving:

```bash
mlx-moe-stream train-predictor \
  --trace routes.jsonl \
  --model-type qwen3_moe \
  --output qwen3-predictor.json

mlx-moe-stream generate \
  --manifest prepared-qwen3/manifest.json \
  --io-workers 1 \
  --prefetch-depth 1 \
  --async-gpu \
  --predictor qwen3-predictor.json \
  --predictive-prefetch-candidates 4 \
  --predictive-min-confidence 0.25 \
  --predictive-prefetch-budget 32MB \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16
```

The predictor's model type, layer count, and expert count must exactly match
the prepared manifest; a mismatch is rejected before inference. M10 also
requires at least one M6 I/O worker. Each router call has independent hard
limits for candidate count and speculative bytes, in addition to the existing
bounded loader queue. CLI logs and server `/metrics` report submitted,
used, and unused predictive reads separately from M6's known-route prefetch.

The included `trace` command currently captures Qwen3-MoE route traces. The
predictor format is family-neutral, but Qwen3.5 and Gemma4 need a matching
family route JSONL before M10 will enable for their manifests.

## Forced-oversubscription decode benchmark (M4)

Run the same prompt at a deliberately small fraction of all expert bytes. Each
JSON result includes decode tok/s, p50/p95 latency, cache and byte hit rates,
disk bytes/token, and evictions/token. Omit `--budget-fraction` for the M3
no-cache baseline.

`disk_bytes` is the exact logical byte range requested from the expert store.
macOS may satisfy repeat `pread()` calls from its page cache, so use a controlled
cold-cache environment before interpreting tok/s as physical-SSD throughput.
M4 synchronizes at the sparse-layer boundary to release pin-safe MLX arrays;
M6 is the milestone that removes this scheduling limitation with explicit I/O
and GPU overlap.

```bash
python benchmarks/decode.py \
  --manifest prepared-qwen3/manifest.json \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 16 \
  --budget-fraction 0.1 \
  --budget-fraction 0.2 \
  --budget-fraction 0.3 \
  --budget-fraction 0.5
```

For an exact reference comparison while cache-enabled:

```bash
python benchmarks/correctness.py \
  --manifest prepared-qwen3/manifest.json \
  --prompt "Hi." \
  --resident-budget 2GB
```

You can run the same trace workflow directly:

```bash
python benchmarks/routing_trace.py --help
```

## Development milestones

1. **Current (M0–M8):** package scaffold, routing trace, LRU simulation,
   manifest, selective reads, exact no-cache execution, a pin-safe global
   byte-budgeted cache, exact expert-major prefill, and bounded I/O/GPU
   overlap with timeline metrics; automatic safe expert budgets and pressure
   protection; a bounded local OpenAI-compatible server with request metrics.
2. **Later:** M9 predictive prefetch and M10 packed-format/Metal optimization.

See [THEORY.md](THEORY.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
[IMPLEMENTATION.md](IMPLEMENTATION.md) for the fixed correctness constraints
and milestone gates.
