# IMPLEMENTATION.md
## Codex 구현 및 검증 계획

**문서 상태**  
Implementation Plan v0.1  
기준일 2026-08-23

이 문서는 Codex가 repository에서 바로 구현을 시작할 수 있도록 작업 순서와 완료 조건을 고정한다.

중요한 원칙

> 성능 최적화보다 먼저 routing trace와 correctness baseline을 만든다.  
> 실제 RAM 초과 모델부터 띄우지 않는다.  
> 인위적 resident budget 제한으로 동일 문제를 재현한 뒤 각 단계의 이득을 측정한다.

---

## 1. 작업 순서

```text
M0 Scaffold
  ↓
M1 Routing trace
  ↓
M2 Disk manifest + selective expert read
  ↓
M3 Streaming MoE correctness
  ↓
M4 Resident cache decode
  ↓
M5 Expert-major prefill
  ↓
M6 I/O overlap
  ↓
M7 Memory budget + metrics
  ↓
M8 Local API server
  ↓
M9 Predictive prefetch
  ↓
M10 ExpertPack / Metal optimization
```

M1~M4를 통과하기 전 M8을 시작하지 않는다.

---

## 2. M0 Repository scaffold

### 구현

- `pyproject.toml`
- `src/` layout
- CLI entry point
- structured logging
- config dataclasses
- pytest
- benchmark directory
- CI 기본 lint/test

working package name

```text
mlx-moe-stream
```

import name

```text
mlx_moe_stream
```

### 초기 dependency 방향

현재 검증 기준

- MLX 0.32.x
- mlx-lm 0.31.x
- safetensors 0.8.x
- Python >= 3.10

정확한 upper/lower bound는 실제 install matrix를 돌린 후 고정한다.

### 완료 조건

```bash
pip install -e .
mlx-moe-stream --help
pytest
```

가 Apple Silicon에서 성공.

---

## 3. M1 Routing trace first

### 목적

아무 cache도 만들지 않고 reference Qwen3-MoE의 route를 측정한다.

### 구현

mlx-lm Qwen3 MoE block에 최소 invasive hook을 추가한다.

수집

```text
request_id
phase         prefill | decode
token_index
layer_id
expert_ids
router_scores
timestamp
```

### 산출 통계

layer별

- num_experts
- top-k
- expert frequency
- route entropy
- consecutive-token Jaccard
- exact overlap count
- unique expert count
- cumulative working-set curve

cache simulator

```text
LRU capacities
5%
10%
20%
30%
50%
100%
```

에 대해

- hit rate
- byte hit rate
- eviction rate

를 offline 계산한다.

### 핵심 판정

working-set 20~30%에서 hit rate가 거의 0에 가까우면 decode cache의 기대효과가 작다는 경고를 출력한다.

### 테스트

- hook가 logits를 바꾸지 않음
- tracing on/off 동일 output
- route index 범위 유효

### 완료 조건

하나의 실제 Qwen3-MoE prompt에서 JSONL trace와 summary table 생성.

---

## 4. M2 Manifest와 selective disk read

### 목적

모델 전체를 MLX로 materialize하지 않고 expert 하나의 exact bytes만 읽을 수 있어야 한다.

### 구현 순서

1. safetensors index 탐색
2. header parser 또는 safetensors metadata API로 tensor shape/dtype/offset 확보
3. Qwen3 expert naming pattern 인식
4. expert bundle manifest 생성
5. `os.pread` exact range loader 구현
6. read byte counter 구현

### 지원 layout

#### split experts

```text
...experts.12.gate_proj.weight
...experts.12.up_proj.weight
...experts.12.down_proj.weight
```

#### leading expert axis

```text
...experts.gate_up_proj
shape [E, ...]
```

두 번째 방식에서는 expert axis 0 slice가 contiguous인지 검증한다.

### Quantized bundle

한 expert key의 read는 필요한 모든 packed tensor와 quantization metadata를 포함해야 한다.

예

```text
gate weight
gate scale
gate bias
up weight
up scale
up bias
down weight
down scale
down bias
```

실제 모델 형식에 존재하는 항목만 포함.

### 테스트

- expert 0 first byte
- expert last byte
- random expert 100개
- source tensor slice와 bitwise 비교
- corrupted offset fail
- truncated file fail

### 반드시 측정

expert 하나를 요청했을 때 disk read counter가 full tensor 크기로 증가하면 실패다.

---

## 5. M3 Streaming Qwen3 MoE correctness

### 목적

standard `SwitchGLU`를 out-of-core reference backend로 교체하되 output을 보존한다.

### 구현

`Qwen3MoeAdapter`

1. config probe
2. non-expert weight load
3. routed expert weight를 model state에서 제외
4. MoE block을 `StreamingSwitchGLU`로 replace
5. router는 기존 mlx-lm 구현 그대로 사용

### ReferenceExpertBackend

속도보다 정확성 우선.

router가 선택한 expert를 하나씩 materialize하고 계산.

가능하면 MLX existing quantized matmul 사용.

### 중요

standard mlx-lm Qwen3 `sanitize()`가 expert 전체를 `mx.stack()`하는 부분은 out-of-core adapter에서 사용하지 않는다.

### correctness test

동일 quantized model을

A. standard mlx-lm full-resident  
B. mlx-moe-stream streaming reference

두 경로로 실행.

비교

- router expert indices exact match
- logits `allclose`
- greedy generated tokens exact match
- multiple prompt lengths
- prefill + decode

### 완료 조건

최소 20개 prompt에서 greedy output mismatch 0.

---

## 6. M4 ResidentCache decode

### 목적

같은 expert를 매번 disk에서 읽지 않게 한다.

### 구현

global byte-budget LRU.

필수 API

```python
cache.get(key)
cache.reserve(nbytes)
cache.admit(expert)
cache.pin(key)
cache.unpin(key)
cache.evict_until(nbytes)
cache.stats()
```

### forced oversubscription

실제 모델이 RAM에 충분히 들어가더라도 다음으로 제한한다.

```text
expert cache = total expert bytes × 0.1
expert cache = total expert bytes × 0.2
expert cache = total expert bytes × 0.3
expert cache = total expert bytes × 0.5
```

non-expert weight는 resident.

### benchmark

각 capacity마다

- tok/s
- p50 token latency
- p95 token latency
- cache hit
- byte hit
- disk bytes/token
- evictions/token

### baseline

`NoCacheBackend`

매 선택 expert를 disk에서 다시 읽는다.

### 완료 조건

route locality가 존재하는 workload에서 LRU가 disk bytes/token과 token latency를 일관되게 줄인다.

---

## 7. M5 Expert-major prefill

### 목적

prefill token별로 expert를 load하는 최악의 구현을 제거한다.

### 구현

layer마다

```text
router(all tokens)
  ↓
expert -> token indices
  ↓
load expert
  ↓
compute selected token group
  ↓
scatter add
```

gating weight와 output shape는 reference와 동일.

### resident 우선

prefill expert order baseline을 세 가지 비교한다.

1. expert id ascending
2. resident experts first
3. disk offset ascending

### benchmark prompt

```text
256
1024
2048
8192 tokens
```

가능한 context 범위에서 수행.

측정

- TTFT
- prefill tok/s
- unique expert union per layer
- disk bytes
- read count
- average read size

### 완료 조건

naive token-major streaming보다 현저히 적은 disk read와 TTFT.

---

## 8. M6 I/O overlap

### 목적

GPU가 expert A를 계산하는 동안 SSD에서 expert B를 읽는다.

### 구현

- bounded I/O ThreadPool
- in-flight Future registry
- demand/prefetch priority
- duplicate load coalescing
- `mx.async_eval`
- 명시적 synchronization timing

### 실험

세 가지 경로

A. sequential read -> compute  
B. threaded prefetch without GPU async  
C. async GPU compute + threaded prefetch

측정

\[
T_A,\ T_B,\ T_C
\]

C가 가장 빨라야 한다.

### timeline logging

```text
load_start
load_end
materialize_start
materialize_end
gpu_enqueue
gpu_done
```

이벤트를 trace file로 남긴다.

### 완료 조건

최소 한 meaningful prefill/decode workload에서 compute/I/O overlap이 profiler로 확인됨.

---

## 9. M7 MemoryBudgetManager

### 구현

startup에서

```python
mx.device_info()
```

를 기록한다.

추적

- memory_size
- max_recommended_working_set_size
- MLX active memory
- MLX peak memory
- process RSS
- swap usage

### auto budget

기본

```text
expert_budget =
safe_working_set
- nonexpert
- KV reserve
- scratch reserve
```

safety margin은 config로 노출.

### memory pressure 단계

1. speculative prefetch off
2. LRU aggressive eviction
3. expert cache shrink
4. request reject with explicit error

swap를 자동 성공 경로로 간주하지 않는다.

### wired memory

`mx.set_wired_limit`은 feature flag.

default OFF.

---

## 10. M8 Local API server

### 목적

Codex 또는 OpenAI SDK가 local model endpoint로 사용할 수 있게 한다.

### endpoint

```text
GET  /health
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
GET  /metrics
```

### 초기 제한

- one active generation
- bounded prompt size
- bounded max_tokens
- localhost default
- no authentication for localhost only

### request metrics

각 request에

```text
TTFT
prefill tok/s
decode tok/s
cache hit
disk GB
peak resident bytes
```

를 internal log에 남긴다.

### 완료 조건

OpenAI Python client의 `chat.completions`로 end-to-end 호출 성공.

---

## 11. M9 Predictive prefetch

M4 결과가 유의할 때만 진행.

### Phase A Last route

\[
\hat S_{t+1,l}=S_{t,l}
\]

### Phase B Transition table

expert별 transition counts.

\[
P(e_j^{t+1}\mid e_i^t)
\]

### admission rule

prefetch candidate는

- 현재 resident 아님
- in-flight 아님
- prefetch budget 이하
- estimated utility threshold 이상

일 때만 read.

### 평가

- precision
- recall
- useful byte ratio
- additional disk bytes
- change in tok/s
- cache pollution

### kill condition

prefetch가 disk bytes만 증가시키고 tok/s를 개선하지 못하면 기본 OFF.

---

## 12. M10 ExpertPack

다음 중 하나가 profiler에서 병목일 때만 구현.

- read request가 지나치게 많음
- 작은 gate/up/down tensor read가 분산됨
- safetensors expert layout이 불리함
- `pread` fixed overhead가 큼

### packer

offline conversion

```bash
mlx-moe-stream repack \
  --model ... \
  --format expert-pack-v1
```

### 목표

한 `(layer, expert)` bundle을 가능한 한 단일 contiguous read로 가져온다.

alignment는 실제 Metal/storage benchmark 후 결정한다.

처음부터 4KB/16KB/64KB 중 하나를 가정하지 않는다.

---

## 13. M11 Custom Metal backend

다음 profile이 확인된 경우만 구현.

```text
Python dispatch / dynamic expert loop
> 10~15% token latency
```

또는

existing MLX primitive가 dynamic resident slot layout을 효율적으로 처리하지 못할 경우.

### 단계

1. single quantized expert GEMV
2. gate/up fusion
3. SwiGLU fusion
4. down projection
5. multi-expert weighted accumulation
6. optional slot indirection

각 단계마다 reference backend와 numerical test.

### 금지

처음부터 전체 MoE block을 하나의 거대한 Metal kernel로 작성하지 않는다.

---

## 14. Semantic cache는 별도 milestone

FreeToken의 semantic-aware recurrent/KV cache는 가치가 크지만, out-of-core expert serving의 핵심 hypothesis와 분리한다.

MVP가 성공한 후 agentic workload용으로 추가한다.

우선순위

1. ordinary prefix KV reuse
2. semantic boundary detection
3. recurrent state snapshot이 필요한 hybrid model adapter
4. cache budget과 expert budget 통합

Qwen3-MoE reference가 ordinary attention 구조라면 initial release blocker가 아니다.

---

## 15. Benchmark matrix

### Hardware metadata

반드시 기록

```text
Mac model
SoC
CPU cores
GPU cores
Unified Memory
macOS
SSD capacity
free SSD
MLX version
mlx-lm version
Python version
```

### Disk microbenchmark

- sequential 1GB
- random expert-sized reads
- 1, 2, 4, 8 I/O workers
- warm page cache
- cold-ish page cache 가능한 범위

### Model scenarios

#### Scenario A Fit

전체 model resident 가능.

목적 correctness와 upper bound.

#### Scenario B Artificial oversubscription

expert cache budget을 전체 expert bytes보다 작게 제한.

목적 실제 RAM이 작은 것처럼 policy 검증.

#### Scenario C Actual over-memory

serialized model + KV + runtime가 safe resident budget보다 큼.

목적 최종 claim 검증.

---

## 16. End-to-end benchmark workloads

### Prompt classes

1. short QA
2. code generation
3. long-context summarization
4. multi-turn agent-like history

### lengths

가능한 모델 context 내에서

```text
prompt 256 / 2K / 8K
generation 128 / 256
```

### repeated prompt

같은 domain prompt를 연속 요청하여 cross-request expert locality도 별도 측정.

초기 server가 single active request여도 sequential request cache는 유지할 수 있다.

---

## 17. Baselines

최소 baseline

### B0 Full resident mlx-lm

가능한 모델에서 upper bound.

### B1 No cache streaming

모든 selected expert reload.

### B2 LRU streaming

본 시스템 core.

### B3 LRU + prefetch

추가 최적화.

### B4 naive MLX lazy / OS pressure

재현 가능할 경우 비교.

외부 llama.cpp 또는 다른 engine은 동일 model/quant format의 공정한 비교가 가능한 경우만 추가한다.

---

## 18. 주요 지표

### Primary

- decode tok/s
- p95 token latency
- TTFT
- disk bytes/token
- cache byte hit rate

### Secondary

- read IOPS
- effective disk bandwidth
- prefetch useful bytes
- expert eviction rate
- process RSS
- swap used
- MLX peak memory

단순 평균 tok/s 하나로 결론 내리지 않는다.

---

## 19. Correctness test matrix

### deterministic

- greedy decoding
- fixed prompt
- same quantized weights
- full-resident reference vs streaming

### numerical

각 MoE layer standalone

```text
same x
same expert ids
same scores
```

output compare.

### prefill

multiple tokens and duplicate expert routes.

### decode

cache hit/miss combination 전부.

### eviction race

in-flight expert eviction 불가 검증.

### failed prefetch

실제 routing에 영향 없음.

---

## 20. Performance regression tests

CI에서 실제 대형 model benchmark는 돌리지 않는다.

microbenchmark fixture를 만든다.

- synthetic expert files
- predictable routing trace
- configurable expert size
- fake slow storage backend

이를 통해

- duplicate load coalescing
- LRU
- byte budget
- prefetch priority
- eviction
- cancellation

을 deterministic하게 검사한다.

실제 Mac benchmark는 별도 script.

---

## 21. Codex가 지켜야 할 구현 규칙

1. `THEORY.md`의 exact inference 원칙을 깨지 않는다.
2. model-specific code를 core cache/storage에 섞지 않는다.
3. storage layer가 MLX model object를 알지 않게 한다.
4. adapter가 disk I/O를 직접 하지 않는다.
5. cache가 tokenizer/server를 알지 않게 한다.
6. benchmark 없는 custom kernel을 추가하지 않는다.
7. hidden automatic fallback을 만들지 않는다.
8. unsupported quantization은 명시적 error.
9. 모든 optimization은 disable 가능한 flag를 둔다.
10. metric으로 효과가 증명되지 않은 optimization은 default ON으로 만들지 않는다.

---

## 22. PR 단위 권장

### PR 1
Scaffold + config + logging

### PR 2
Qwen3 route tracing + cache simulator

### PR 3
Safetensors manifest + selective read

### PR 4
Streaming Qwen3 reference backend + correctness

### PR 5
Resident LRU + forced oversubscription benchmark

### PR 6
Expert-major prefill

### PR 7
Async I/O overlap

### PR 8
Memory budget + metrics

### PR 9
Local OpenAI-compatible server

### PR 10
Prefetch experiments

### PR 11+
ExpertPack / Metal only if profiler warrants

각 PR은 독립 benchmark 또는 correctness evidence를 포함해야 한다.

---

## 23. 최초 Codex 실행 지시

repository가 비어 있다면 Codex는 다음까지만 먼저 수행한다.

```text
1. 문서 세 개 읽기
2. pyproject와 src/test scaffold 생성
3. Qwen3-MoE route trace hook 구현
4. route trace unit/integration test
5. offline LRU simulator 구현
6. benchmark/routing_trace.py 작성
7. README에 실행법 작성
```

이 단계에서는 아직

- streaming expert load
- custom Metal
- server
- packed format

을 구현하지 않는다.

M1 결과로 route locality와 예상 cache hit curve를 먼저 본다.

---

## 24. Go / No-Go decision

### Go

다음이 관찰되면 M2 이후 진행.

- route trace 정상
- small working set에서 non-trivial hit rate
- Qwen3 expert tensor layout이 selective read 가능
- selected expert bundle 크기가 NVMe bandwidth 대비 현실적

### Re-evaluate

- route가 거의 iid uniform
- expert bundle 하나가 지나치게 큼
- quantized source가 slicing 불가능
- model adapter가 expert 전체 materialization 없이 구성 불가능

이 경우 model family 또는 storage format을 먼저 바꾼다.

---

## 25. 공개 패키지 최소 완료 기준

`pip install mlx-moe-stream` 이후 사용자가

```bash
mlx-moe-stream prepare --model ...
mlx-moe-stream serve --model ...
```

로 실행 가능.

문서화된 적어도 하나의 Qwen3-MoE configuration에서

- full-resident reference와 correctness 일치
- configurable resident budget 작동
- cache와 exact disk streaming 작동
- metrics 제공
- OpenAI chat endpoint 작동

그리고 적어도 하나의 over-budget 실험에서

- naive streaming 또는 OS oversubscription보다 명백히 빠름
- sustained decode가 practical threshold에 도달

해야 한다.

---

## 26. 현재 기술적 위험

### Risk A MLX materialization copy

disk bytes를 MLX array로 만들 때 copy와 allocation이 병목이 될 수 있다.

대응

- measure first
- resident object reuse
- 이후 native extension/Metal buffer path 검토

### Risk B random I/O

expert bundle이 여러 shard에 흩어져 read IOPS가 증가할 수 있다.

대응

- manifest statistics
- ExpertPack

### Risk C prefill explosion

긴 prompt는 모든 expert를 건드린다.

대응

- expert-major grouping
- read/compute pipeline
- TTFT 별도 관리

### Risk D cache pollution from predictor

대응

- prefetch budget
- useful byte metric
- default OFF until validated

### Risk E Unified Memory pressure

대응

- recommended working set 기준
- resident budget margin
- speculative cache 우선 해제
- swap을 success path로 간주하지 않음

### Risk F mlx-lm internal API change

adapter가 private implementation에 과도하게 의존하면 쉽게 깨진다.

대응

- version compatibility layer
- Qwen3 adapter contract test
- known-good version matrix

---

## 27. 이 문서가 완료되었다고 볼 조건

Codex가 추가 설계 결정을 임의로 만들지 않고 다음 질문에 답할 수 있어야 한다.

- 어디를 hook해야 하는가
- 어떤 weight를 resident로 남기는가
- expert는 어떤 key로 관리하는가
- disk에서 무엇을 몇 byte 읽는가
- cache miss는 어떻게 처리하는가
- prefill과 decode가 왜 다른가
- 정확성은 무엇과 비교하는가
- 어떤 benchmark가 다음 milestone 진입을 허용하는가
- 언제 custom Metal을 도입하는가
- 실패하면 어떤 가정을 재검토하는가

이 기준을 유지하면서 구현을 진행한다.
