# ARCHITECTURE.md
## `mlx-moe-stream` 시스템 아키텍처

**문서 상태**  
Design Spec v0.1  
기준일 2026-08-23

---

## 1. 패키지 역할

`mlx-moe-stream`은 mlx-lm을 대체하는 새 LLM ecosystem이 아니다.

역할은 다음처럼 제한한다.

> mlx-lm의 model architecture, tokenizer, attention, KV cache, sampling을 최대한 재사용하고, MoE expert weight residency와 execution path만 out-of-core aware하게 교체한다.

초기 지원 플랫폼

- Apple Silicon
- macOS
- Python 3.10+
- MLX
- mlx-lm
- local single-user serving

초기 reference family

- Qwen3-MoE
- smoke test는 Qwen3-30B-A3B 계열
- 실제 over-memory 검증은 별도 대형 MoE 또는 인위적 resident budget 축소로 수행

---

## 2. 공개 UX

### 설치

```bash
pip install mlx-moe-stream
```

### 모델 준비

```bash
mlx-moe-stream prepare \
  --model <hf-repo-or-local-path> \
  --output ~/.cache/mlx-moe-stream/<model>
```

### CLI generation

```bash
mlx-moe-stream generate \
  --model ~/.cache/mlx-moe-stream/<model> \
  --prompt "Explain the result." \
  --resident-budget 32GB
```

### local server

```bash
mlx-moe-stream serve \
  --model ~/.cache/mlx-moe-stream/<model> \
  --host 127.0.0.1 \
  --port 8080 \
  --resident-budget auto
```

초기 서버는 local endpoint만 목표로 한다.

---

## 3. 전체 구조

```text
                    User / OpenAI client
                             |
                             v
                      Local API Server
                             |
                             v
                       GenerationEngine
                             |
              +--------------+--------------+
              |                             |
              v                             v
       mlx-lm attention               MoE Runtime
       tokenizer / KV                 replacement
                                            |
                              +-------------+-------------+
                              |             |             |
                              v             v             v
                         RouterTrace   ResidentCache  Prefetcher
                              |             |             |
                              +-------------+-------------+
                                            |
                                            v
                                       ExpertStore
                                            |
                                +-----------+-----------+
                                |                       |
                                v                       v
                         Safetensors spans        ExpertPack v1
                                |                       |
                                +-----------+-----------+
                                            |
                                            v
                                          NVMe
```

execution data path는 다음과 같다.

```text
router
  |
  v
ExpertKey(layer, expert)
  |
  +--> resident hit --------------------+
  |                                     |
  +--> miss -> pread -> materialize ----+--> ExpertBackend --> GPU
```

---

## 4. 핵심 설계 원칙

1. **non-expert weight는 기존 mlx-lm 경로를 사용**
2. **expert weight만 별도 ExpertStore에서 관리**
3. standard mlx-lm의 Qwen3 `sanitize()`가 모든 expert를 `mx.stack()`하는 경로를 우회
4. model output semantics는 유지
5. cache는 `(layer_id, expert_id)` 단위
6. cache capacity는 slot count가 아니라 bytes
7. disk read 범위가 명시적이어야 함
8. OS swap은 fallback이지 scheduler가 아님
9. prefetch는 speculative, execution은 exact
10. custom Metal kernel은 MVP 이후

---

## 5. Repository 구조

Codex는 최초 repository를 다음 구조로 만든다.

```text
mlx-moe-stream/
├── pyproject.toml
├── README.md
├── THEORY.md
├── ARCHITECTURE.md
├── IMPLEMENTATION.md
├── src/
│   └── mlx_moe_stream/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── runtime.py
│       ├── memory.py
│       ├── metrics.py
│       ├── manifest.py
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── safetensors_store.py
│       │   └── expert_pack.py
│       ├── cache/
│       │   ├── __init__.py
│       │   ├── resident.py
│       │   └── policy.py
│       ├── prefetch/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── last_route.py
│       │   └── transition.py
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── reference.py
│       │   └── metal.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   └── qwen3_moe.py
│       └── server/
│           ├── __init__.py
│           └── app.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── correctness/
└── benchmarks/
    ├── routing_trace.py
    ├── storage_read.py
    ├── decode.py
    ├── prefill.py
    └── end_to_end.py
```

MVP 시점에는 `metal.py`와 `expert_pack.py`가 stub이어도 된다.

---

## 6. Core data model

### 6.1 ExpertKey

```python
@dataclass(frozen=True, order=True)
class ExpertKey:
    layer: int
    expert: int
```

global cache의 identity다.

---

### 6.2 TensorSpan

```python
@dataclass(frozen=True)
class TensorSpan:
    file: Path
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    role: str
```

`role` 예

- `gate_weight`
- `gate_scale`
- `up_weight`
- `up_scale`
- `down_weight`
- `down_scale`
- `bias`

---

### 6.3 ExpertBundleSpec

```python
@dataclass(frozen=True)
class ExpertBundleSpec:
    key: ExpertKey
    tensors: tuple[TensorSpan, ...]
    total_bytes: int
    quantization: QuantizationSpec
```

한 expert를 실행하는 데 필요한 disk data 전체를 의미한다.

---

### 6.4 ResidentExpert

```python
@dataclass
class ResidentExpert:
    key: ExpertKey
    arrays: dict[str, mx.array]
    nbytes: int
    last_used_step: int
    pin_count: int = 0
```

`pin_count > 0`인 expert는 현재 in-flight 계산 중이므로 eviction하면 안 된다.

---

## 7. Manifest

`prepare` 단계는 runtime이 model repository 구조를 반복 분석하지 않도록 manifest를 생성한다.

예시

```json
{
  "format_version": 1,
  "model_type": "qwen3_moe",
  "source_model": "...",
  "num_layers": 48,
  "num_experts": 128,
  "experts_per_token": 8,
  "quantization": {
    "bits": 4,
    "group_size": 64
  },
  "non_expert_weight_files": [...],
  "expert_bundles": {
    "0:0": {...},
    "0:1": {...}
  }
}
```

manifest 생성 시 반드시 다음을 검증한다.

- file 존재
- offset 범위
- dtype
- expected shape
- expert axis
- quantization metadata
- 모든 MoE layer의 expert count 일치 여부
- bundle마다 필요한 projection/scale 존재 여부

runtime에서 발견하는 대신 prepare 단계에서 fail-fast한다.

---

## 8. Safetensors storage backend

### 8.1 왜 `mx.load(full_shard)`를 사용하지 않는가

standard mlx-lm load path는 lazy mode가 있어도 결국 필요한 tensor를 MLX array로 materialize한다.

out-of-core runtime은 expert 하나를 읽기 위해 expert axis 전체 tensor가 resident가 되는 경로를 피해야 한다.

따라서 MVP storage backend는 disk byte range를 명시적으로 관리한다.

---

### 8.2 read 방식

우선순위

1. safetensors header에서 offset과 shape 분석
2. expert가 contiguous한 byte range인지 확인
3. `os.pread()`로 exact range read
4. CPU buffer에서 dtype view 생성
5. MLX array로 materialize
6. `mx.async_eval()`로 GPU execution 전에 준비

`pread`를 사용하는 이유

- shared file position이 없음
- 여러 I/O worker에서 안전
- exact range 제어
- random expert read 측정이 쉬움

safetensors 0.8 계열에는 pread backend도 존재하지만, package의 핵심 추상화는 특정 safetensors Python API에 묶지 않는다.

---

## 9. Qwen3 adapter

현재 mlx-lm의 Qwen3 MoE는 router 이후 `SwitchGLU`를 호출하며, standard `sanitize()`는 expert weight를 expert axis로 `mx.stack()`한다.

out-of-core adapter는 이 부분을 변경한다.

### standard path

```text
HF expert tensors
     |
sanitize()
     |
mx.stack(all experts)
     |
SwitchGLU
     |
gather_qmm
```

### streaming path

```text
HF expert tensors
     |
manifest only
     |
StreamingSwitchGLU
     |
ExpertRuntime.resolve(layer, expert_ids)
     |
resident arrays
     |
ExecutionBackend
```

중요한 점은 model architecture 전체를 fork하지 않고 **MoE block만 adapter로 교체**하는 것이다.

---

## 10. ModelAdapter protocol

```python
class ModelAdapter(Protocol):
    def probe(self, config: dict) -> bool: ...
    def build_manifest(self, model_path: Path) -> ModelManifest: ...
    def load_shell(self, model_path: Path, manifest: ModelManifest): ...
    def replace_moe_blocks(self, model, runtime): ...
```

`load_shell`은

- attention
- embedding
- router gate
- norm
- shared expert
- lm_head

등 non-routed weight만 정상 load한다.

routed expert weight는 model module에 전체 tensor로 붙이지 않는다.

---

## 11. ResidentCache

### 11.1 정책

global byte-budget LRU.

```text
key = (layer, expert)
value = ResidentExpert
capacity = resident_expert_budget_bytes
```

invariant

```text
resident_bytes <= expert_budget
```

새 expert admission이 필요하면

1. 현재 사용 중인 pinned expert 제외
2. LRU 순서로 eviction
3. 필요한 byte 확보
4. 새 expert materialize

---

### 11.2 cache metrics

반드시 측정

- lookup count
- hit count
- miss count
- byte hit rate
- eviction count
- admission bytes
- reload bytes
- average residence duration
- per-layer hit rate

token count 기반 hit rate보다 byte hit rate가 더 중요하다.

---

## 12. MemoryBudgetManager

입력

- `mx.device_info()`
- physical memory
- max recommended working set
- 사용자 `--resident-budget`
- 현재 non-expert measured bytes
- KV reserve
- scratch reserve

출력

- expert cache budget
- maximum KV budget
- warning level

초기 auto policy

```text
safe_budget
  = min(user_limit, recommended_working_set)
  - OS_safety_margin

expert_budget
  = safe_budget
  - nonexpert_bytes
  - kv_reserve
  - scratch_reserve
```

`mx.set_wired_limit()`은 opt-in이다.

기본 install 또는 serve에서 system `sysctl`을 변경하지 않는다.

---

## 13. Decode Scheduler

layer별 algorithm

```text
1 router computes ids and scores
2 cache.resolve(ids)
3 split hits / misses
4 submit misses to IO
5 enqueue hit expert compute
6 await materialization of misses
7 enqueue miss expert compute
8 weighted merge
9 update cache recency
10 emit route metrics
```

decode batch 1을 최초 최적화 target으로 한다.

multi-request dynamic batching은 initial scope에서 제외한다.

---

## 14. ExecutionBackend

### 14.1 Reference backend

목표는 correctness다.

expert별로

- quantized matmul 또는 equivalent MLX op
- SwiGLU
- down projection
- gating score multiply

를 수행한다.

Python loop가 느려도 괜찮다.

이 backend가 golden reference가 된다.

---

### 14.2 Gather backend

resident expert를 small contiguous group으로 만들어 existing MLX `gather_qmm`을 활용할 수 있는지 benchmark한다.

단, resident cache update 때 전체 bank copy가 발생하면 이 경로는 버린다.

---

### 14.3 Custom Metal backend

성능 병목이 Python dispatch 또는 dynamic expert indirection이면 custom Metal kernel을 구현한다.

목표 interface

```text
input
  hidden
  expert_slot_ids
  router_scores
  packed slot banks

kernel
  indexed quantized GEMV
  activation
  down projection
  weighted accumulation
```

처음부터 fused mega-kernel을 만들지 않는다.

최소 kernel 단위로 benchmark 후 fusion한다.

---

## 15. Prefill Scheduler

prefill은 다음 algorithm을 사용한다.

```text
for each MoE layer
    run router for all current tokens
    build expert -> token_indices map

    order experts
    for expert in order
        prefetch next expert
        materialize current expert
        run current expert for grouped tokens
        scatter weighted output
```

expert ordering baseline

- expert id ascending

향후

- current resident first
- disk physical offset order
- co-activation packed order

를 비교한다.

---

## 16. I/O subsystem

MVP

- fixed-size ThreadPoolExecutor
- `os.pread`
- per-file descriptor reuse
- bounded queue
- duplicate request coalescing

필수 기능

### Request coalescing

동일 `(layer, expert)`가 동시에 request되면 disk read는 한 번만 한다.

```text
MISS A
MISS A
MISS A
   |
single Future[ResidentExpert]
```

### Backpressure

prefetch가 demand miss를 밀어내면 안 된다.

priority

1. exact demand miss
2. next-needed prefill load
3. speculative decode prefetch

---

## 17. Prefetcher

interface

```python
class Prefetcher(Protocol):
    def observe(self, trace: RouteTrace) -> None: ...
    def predict(self, context: PrefetchContext) -> list[ExpertKey]: ...
```

초기 구현

### NoPrefetch

benchmark baseline.

### LastRoutePrefetch

이전 token 동일 layer route를 next-token candidate로 사용.

### TransitionPrefetch

online transition count

\[
P(e'|e)
\]

기반.

prefetch는 expert cache budget의 별도 비율을 초과할 수 없다.

예

```text
prefetch_admission_budget <= 20% of expert cache
```

초기값은 실험으로 조정한다.

---

## 18. ExpertPack v1

MVP는 source safetensors에서 직접 읽는다.

성능 프로파일에서 random read 및 작은 span overhead가 크면 offline packed format을 추가한다.

목표

```text
ExpertPack
├── manifest.json
├── nonexpert/
└── experts/
    ├── layer_000.pack
    ├── layer_001.pack
    └── ...
```

각 layer pack에서 expert bundle을 page-aligned contiguous region으로 둔다.

```text
[expert 0 bundle][padding]
[expert 1 bundle][padding]
...
```

장점

- read request 감소
- gate/up/down scale을 한 I/O로 묶기 쉬움
- direct offset 계산 단순
- sequential expert-major prefill 최적화
- future native Metal buffer alignment 고려 가능

packing format은 benchmark가 필요성을 입증하기 전까지 구현하지 않는다.

---

## 19. MLX streams

MLX는 CPU와 GPU가 같은 Unified Memory array를 사용할 수 있고 stream dependency를 관리한다.

runtime은 별도 GPU stream 사용 여부를 실험하되, 초기 correctness 경로는 default GPU stream을 사용한다.

I/O는 MLX CPU operation이 아니라 OS file I/O thread로 수행한다.

expected overlap

```text
I/O thread       read expert B ---------
GPU stream     compute expert A --------
```

timing 검증 시 `mx.synchronize()` 위치를 명시적으로 관리한다.

---

## 20. KV cache

MVP에서는 mlx-lm cache implementation을 그대로 사용한다.

단 memory budget manager는 KV 사용량을 expert cache와 별도로 추적해야 한다.

초기에는 runtime 도중 expert/KV budget을 자동 resize하지 않는다.

Phase 2 이후 safe point에서

```text
KV grows
  ->
expert budget shrink
  ->
evict LRU experts
```

를 지원한다.

semantic-aware prefix/recurrent state cache는 후속 milestone이다.

---

## 21. Server

기존 `mlx_lm.server`의 OpenAI-like interface를 참고하되 그대로 production server로 간주하지 않는다.

초기 endpoint

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
GET  /metrics
```

초기 concurrency

```text
max_active_generations = 1
```

로 제한한다.

out-of-core expert cache의 correctness와 latency가 안정된 뒤 concurrency를 확장한다.

---

## 22. Observability

모든 benchmark 및 server request에서 다음을 기록한다.

### Memory

- physical memory
- recommended working set
- MLX active memory
- MLX peak memory
- expert resident bytes
- KV estimate
- process RSS
- system swap used

### I/O

- bytes read
- reads count
- average read size
- p50/p95 read latency
- effective MB/s
- demand bytes
- prefetch bytes
- wasted prefetch bytes

### Routing

- expert calls
- unique experts per layer
- previous-token overlap
- entropy
- cache hit rate
- byte hit rate

### Inference

- model load time
- TTFT
- prompt tok/s
- decode tok/s
- token p50/p95 latency

---

## 23. Correctness invariants

1. router result는 reference와 동일
2. expert weights는 source model과 동일한 quantized representation
3. miss load가 완료되기 전 해당 expert output을 만들지 않음
4. eviction 중인 expert는 in-flight compute에서 사용되지 않음
5. prefetch 결과는 routing decision을 바꾸지 않음
6. request abort 시 pin count와 future 정리
7. OOM 위험 시 cache admission을 거부하거나 budget을 축소
8. invalid manifest로 추론 시작 금지

---

## 24. Thread safety

초기 single-generation이어도 I/O worker 때문에 state synchronization이 필요하다.

cache state 변경은 runtime event loop 또는 하나의 scheduler thread에서만 한다.

I/O worker는

```text
bytes -> decoded host representation
```

까지만 반환하고 cache dictionary를 직접 수정하지 않는다.

동일 expert in-flight load는 Future registry로 deduplicate한다.

---

## 25. 예외 처리

### disk read error

request fail. silent fallback 금지.

### memory pressure

- speculative prefetch 중지
- cache budget shrink
- LRU eviction
- 그래도 부족하면 명시적 `MemoryBudgetError`

### unsupported model

`prepare` 단계에서 fail.

### unsupported quantization

runtime에서 즉석 dequant/quant fallback을 만들지 말고 fail-fast.

---

## 26. 보안

- remote model code는 기본적으로 `trust_remote_code=False`
- manifest offset은 file size 범위 검증
- safetensors header size 제한
- path traversal 방지
- local server 기본 bind는 `127.0.0.1`
- 인증 없는 server를 `0.0.0.0`에 자동 노출하지 않음
- prepare가 임의 Python을 실행하지 않도록 함

---

## 27. API 초안

### Python

```python
from mlx_moe_stream import load

engine = load(
    "mlx-community/Qwen3-30B-A3B-4bit",
    resident_budget="24GB",
)

out = engine.generate(
    "Write a short explanation.",
    max_tokens=256,
)
```

### lower-level

```python
engine.stats()
engine.cache.stats()
engine.memory.snapshot()
engine.route_trace.export(...)
```

API는 MVP 결과가 안정되기 전 semantic version 1.0으로 고정하지 않는다.

---

## 28. Non-goals

초기 release에서 하지 않는다.

- training
- fine-tuning
- tensor parallel
- multi-Mac cluster
- CUDA
- Linux
- multi-user production serving
- continuous batching
- approximate expert routing
- custom quantizer zoo
- every mlx-lm model support
- automatic system sysctl tuning

---

## 29. 참고 구현 포인트

현재 mlx-lm Qwen3 MoE는 router에서 top-k expert index를 만들고 `SwitchGLU`로 전달한다. `SwitchGLU`의 quantized path는 `mx.gather_qmm`을 사용한다.

이 구조는 adapter hook로 적합하지만 standard `sanitize()`가 expert tensors를 하나의 stacked array로 변환하므로 out-of-core path에서는 해당 부분을 우회해야 한다.

MLX는 custom Metal kernel을 Python API에서 JIT compile할 수 있으므로, reference backend가 정확성을 확보한 뒤 indexed quantized GEMV 병목을 native kernel로 내릴 수 있다.

