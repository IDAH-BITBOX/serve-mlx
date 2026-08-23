# THEORY.md
## MLX 기반 Out-of-Core MoE Serving 이론 설계

**문서 상태**  
Design Spec v0.1  
기준일 2026-08-23  
작업명 `mlx-moe-stream`  
초기 기준 모델 Qwen3-MoE 계열  
개발 환경 Apple Silicon + MLX + mlx-lm

---

## 1. 목표

이 프로젝트의 목표는 Apple Silicon에서 **물리 메모리 또는 안전한 resident working-set을 초과하는 총 파라미터 규모의 MoE 모델을 정확한 추론 semantics를 유지한 채 실용적인 속도로 서빙**하는 것이다.

핵심 목표는 다음과 같다.

1. 전체 expert pool을 항상 메모리에 상주시킬 필요가 없게 한다.
2. 현재 필요한 expert만 SSD에서 Unified Memory resident working set으로 읽는다.
3. decode에서는 token 간 expert routing locality를 이용해 expert working set을 유지한다.
4. prefill에서는 router 결과를 이용해 expert-major 순서로 필요한 expert를 한 번씩 읽고 다수 token을 처리한다.
5. I/O와 GPU 계산을 최대한 겹친다.
6. prefetch가 틀려도 출력이 변하지 않게 한다.
7. `pip install` 가능한 일반 패키지 형태로 mlx-lm 위에 얹는다.
8. 성능 가정이 틀린 경우를 측정으로 즉시 판정할 수 있게 한다.

이 프로젝트는 FreeToken의 직접 포팅이 아니다. FreeToken의 핵심 원리인 **expert locality, cache, I/O overlap, elastic memory budgeting, state reuse**를 Apple Silicon의 Unified Memory 구조에 맞게 다시 정의하는 프로젝트다.

---

## 2. 핵심 전제와 FreeToken과의 차이

FreeToken은 discrete GPU 환경을 전제로 한다.

\[
\text{Host RAM} \rightarrow \text{PCIe} \rightarrow \text{GPU VRAM}
\]

Apple Silicon에서는 CPU와 GPU가 같은 Unified Memory pool을 직접 사용한다.

\[
\text{CPU} \leftrightarrow \text{Unified Memory} \leftrightarrow \text{GPU}
\]

따라서 FreeToken의 핵심 decode 정책인 PCIe cache fill 대 CPU expert execution의 \(q^\star\) 분할을 그대로 가져오는 것은 적절하지 않다.

이 프로젝트의 memory hierarchy는 다음처럼 정의한다.

\[
\boxed{
\text{SSD}
\rightarrow
\text{pageable Unified Memory}
\rightarrow
\text{resident expert working set}
\rightarrow
\text{GPU execution}
}
\]

실제로는 pageable UM과 resident UM이 물리적으로 다른 메모리가 아니다. 여기서 tier는 **물리 위치가 아니라 residency와 lifetime을 runtime이 얼마나 적극적으로 관리하는지**를 의미한다.

FreeToken에서 VRAM cache의 목적이 PCIe 전송 제거라면, 본 프로젝트에서 resident expert cache의 목적은 **SSD read와 OS swap thrashing을 제거**하는 것이다.

---

## 3. 왜 MoE에서 가능한가

Dense 모델은 매 token마다 대부분의 layer weight를 읽어야 한다. 따라서 모델이 RAM보다 크면 disk bandwidth가 거의 그대로 token latency 상한을 결정한다.

MoE에서는 layer \(l\)에 \(E_l\)개의 routed expert가 있어도 한 token은 \(k_l \ll E_l\)개만 사용한다.

\[
S_{t,l} \subseteq \{1,\dots,E_l\}, \qquad |S_{t,l}|=k_l
\]

전체 파라미터는

\[
P_{\text{total}}
\]

이지만 token \(t\)가 실제로 읽는 routed parameter는

\[
P_{\text{active},t}
=
\sum_l\sum_{e\in S_{t,l}} P_{l,e}
\]

이다.

따라서 핵심 조건은

\[
P_{\text{active}} \ll P_{\text{total}}
\]

이다.

FreeToken 논문 역시 decode에서 인접 token의 expert routing overlap을 활용하는 shared LRU cache를 핵심 구성요소로 사용한다. 본 프로젝트는 이를 SSD-backed working-set 문제로 재해석한다.

---

## 4. 정확성 원칙

이 프로젝트의 기본 모드는 **exact inference**다.

다음은 허용하지 않는다.

- router top-k 축소
- 일부 expert skipping
- approximate expert replacement
- routing predictor 결과를 실제 routing에 사용
- cache miss 시 이전 expert output 재사용
- prefetch 실패 시 근사치 반환

prefetch와 cache는 오직 성능 최적화다.

실제 router가 선택한 expert set을 \(S_{t,l}\), predictor가 예측한 set을 \(\hat S_{t,l}\)라고 하자.

\[
\hat S_{t,l} \neq S_{t,l}
\]

이어도 최종 계산은 반드시

\[
S_{t,l}
\]

에 대해서 수행한다.

즉 predictor의 recall이 낮으면 느려질 뿐, 모델 출력 semantics는 달라지지 않는다.

---

## 5. 기본 비용 모델

### 5.1 정의

layer \(l\), expert \(e\)의 disk-resident bundle 크기를

\[
W_{l,e}
\]

bytes라고 한다.

여기에는 해당 expert 계산에 필요한 모든 tensor가 포함된다.

예

- gate projection
- up projection
- down projection
- quantization scales
- quantization biases 또는 zero points

현재 resident cache를

\[
C_t
\]

라고 한다.

decode token \(t\)의 miss set은

\[
M_{t,l} = S_{t,l}\setminus C_t
\]

이다.

해당 token에서 disk에서 실제 읽어야 하는 양은

\[
D_t
=
\sum_l
\sum_{e\in M_{t,l}}
W_{l,e}
\]

이다.

---

### 5.2 단순한 I/O latency 하한

유효 SSD bandwidth를 \(B_D\), 한 expert bundle read의 평균 fixed overhead를 \(\alpha\), read request 수를 \(n_t\)라고 하면

\[
T_{\text{IO},t}
\approx
n_t\alpha + \frac{D_t}{B_D}
\]

이다.

random read가 많으면 \(\alpha\)와 실제 \(B_D\)가 악화된다. 따라서 단순 LRU뿐 아니라 **expert bundle을 연속된 범위로 저장하고 co-activation이 높은 expert를 인접 배치하는 것**도 중요하다.

---

### 5.3 compute와 I/O overlap

GPU compute time을 \(T_{\text{GPU},t}\), I/O 중 계산 뒤에 숨지 못한 시간을 \(T_{\text{IO,unhidden},t}\)라 하면

\[
T_{\text{token}}
\gtrsim
T_{\text{GPU},t}
+
T_{\text{IO,unhidden},t}
+
T_{\text{sched},t}
\]

이다.

이론적 최선은 read를 모두 숨기는 경우다.

\[
T_{\text{IO,unhidden},t}\rightarrow 0
\]

실제로는 miss가 발생한 현재 layer의 expert는 routing을 본 뒤에야 정확히 알 수 있으므로 완전한 overlap은 불가능하다. 따라서 세 가지 경로가 필요하다.

1. **resident hit**
2. **현재 routing을 본 뒤 즉시 load하는 exact miss**
3. **이전 routing trace 기반 speculative prefetch**

---

## 6. cache hit rate가 가장 중요한 이유

평균 expert bundle 크기를 \(\bar W\), token당 총 routed expert 호출 수를

\[
K=\sum_l k_l
\]

이라고 하자.

byte-weighted hit rate를 \(h\)라 하면 대략

\[
D_{\text{decode}}
\approx
(1-h)K\bar W
\]

이다.

target decode rate를 \(r\) token/s라 하면 token budget은

\[
T_{\text{budget}}=\frac1r
\]

이다.

compute 및 scheduler가 쓰는 시간을 \(T_C\)라고 할 때, disk miss가 만족해야 할 대략적인 조건은

\[
(1-h)K\bar W
\lesssim
B_D
\left(
\frac1r-T_C
\right)
\]

이다.

이 식이 중요한 이유는 **"모델 총 크기가 몇 B인가"보다 "token당 cache 밖에서 몇 byte를 새로 읽는가"가 out-of-core decode 속도를 더 직접적으로 결정**하기 때문이다.

---

## 7. cache 정책

초기 정책은 byte-budget global LRU다.

key는

\[
(l,e)
\]

이다.

전체 cache byte budget을 \(M_E\)라 두면

\[
\sum_{(l,e)\in C_t} W_{l,e}\le M_E
\]

를 항상 만족한다.

단순 slot count가 아니라 byte budget으로 잡아야 한다. 모델마다 expert bundle 크기가 동일하지 않을 수 있기 때문이다.

### 7.1 초기 정책

score는 recency 하나만 사용한다.

\[
score_{l,e} = \text{last\_used\_step}(l,e)
\]

eviction은 가장 오래 사용되지 않은 expert부터 한다.

### 7.2 후속 정책

routing trace가 충분히 쌓이면 expected saved I/O per byte를 사용할 수 있다.

\[
V_{l,e}
=
\frac{
p_{l,e}^{\text{reuse}}\cdot T_{\text{reload},l,e}
}{
W_{l,e}
}
\]

또는

\[
V_{l,e}
=
\frac{
p_{l,e}^{\text{reuse}}
}{
W_{l,e}
}
\]

를 admission 또는 eviction score에 포함한다.

중요한 원칙은 **처음부터 복잡한 predictor를 만들지 않는 것**이다. LRU baseline이 반드시 먼저 존재해야 한다.

---

## 8. decode 실행 모델

token \(t\), layer \(l\)에서 router가 \(S_{t,l}\)를 반환하면 다음 순서로 수행한다.

1. resident hit와 miss를 분리
2. miss read를 즉시 I/O worker에 submit
3. hit expert의 계산을 먼저 GPU에 enqueue
4. miss가 완료되면 resident cache에 materialize
5. miss expert 계산
6. router weight를 적용해 정확히 merge
7. recency 갱신

개념적으로

\[
Y
=
\sum_{e\in H} w_e f_e(X)
+
\sum_{e\in M} w_e f_e(X)
\]

이다.

hit와 miss 계산 순서는 달라도 최종 expert 집합과 gating weight는 동일해야 한다.

floating-point reduction order 차이로 bitwise identity는 보장되지 않을 수 있으므로 correctness gate는 dtype별 tolerance를 사용한다.

---

## 9. predictive prefetch

정확한 다음 token routing은 현재 hidden state가 완성되기 전에 알 수 없다. 따라서 exact lookahead는 없다.

초기 predictor는 다음 정도만 허용한다.

\[
P(e_{t+1,l}\mid e_{t,l})
\]

또는 단순히

\[
\hat S_{t+1,l}=S_{t,l}
\]

이다.

더 복잡한 모델은 routing trace에서 단순 baseline보다 충분히 유의한 이득이 확인된 뒤에만 도입한다.

prefetch 평가 지표는 다음과 같다.

### Precision

\[
\text{precision}
=
\frac{
|\hat S\cap S|
}{
|\hat S|
}
\]

### Recall

\[
\text{recall}
=
\frac{
|\hat S\cap S|
}{
|S|
}
\]

### Useful bytes ratio

\[
U
=
\frac{
\text{prefetched bytes actually used before eviction}
}{
\text{total prefetched bytes}
}
\]

실제 시스템에서는 precision보다 **useful bytes ratio**가 더 중요하다.

---

## 10. prefill은 decode와 다른 문제다

긴 prompt의 prefill에서는 여러 token의 route union이 대부분의 expert를 덮을 수 있다.

layer \(l\)의 prefill token 집합이 \(T\)일 때

\[
U_l
=
\bigcup_{t\in T}S_{t,l}
\]

이다.

긴 prompt에서는

\[
|U_l|\rightarrow E_l
\]

이 될 수 있다.

따라서 decode의 LRU만으로 prefill을 해결할 수 없다.

---

## 11. Apple Silicon용 prefill 전략

FreeToken은 next layer의 전체 expert를 PCIe로 double buffering한다.

본 프로젝트의 MVP에서는 **router-first expert-major execution**을 사용한다.

layer \(l\)에서

1. router를 모든 prompt token에 실행
2. 실제 사용 expert union \(U_l\) 계산
3. token을 expert별로 grouping
4. expert \(e\) bundle을 SSD에서 read
5. 해당 expert를 선택한 모든 token을 한 번에 계산
6. output buffer에 scatter-add
7. 다음 expert를 처리
8. expert \(e+1\) read와 expert \(e\) GPU 계산을 overlap

이 방식의 I/O는

\[
D_{\text{prefill},l}
=
\sum_{e\in U_l}W_{l,e}
\]

이다.

짧은 prompt에서는 full-layer streaming보다 적게 읽을 수 있고, 긴 prompt에서 \(U_l\approx E_l\)이면 full layer read와 같은 수준으로 수렴한다.

중요한 차이는 **한 layer 전체 expert를 동시에 resident로 둘 필요가 없다는 점**이다.

---

## 12. prefill pipeline의 하한

expert-major 순서가 \(e_1,\dots,e_n\)일 때 이상적으로

\[
T_{\text{prefill},l}
\approx
T_{\text{load}}(e_1)
+
\sum_{i=1}^{n-1}
\max
\left(
T_{\text{GPU}}(e_i),
T_{\text{load}}(e_{i+1})
\right)
+
T_{\text{GPU}}(e_n)
\]

이 된다.

따라서 SSD streaming이 유리하려면 평균적으로

\[
T_{\text{load}}(e_{i+1})
\lesssim
T_{\text{GPU}}(e_i)
\]

에 가까울수록 좋다.

그렇지 못하더라도 naive sequential load-then-compute보다는 일부 overlap을 얻을 수 있다.

---

## 13. Unified Memory에서 CPU/GPU hybrid compute를 기본 채택하지 않는 이유

MLX는 CPU와 GPU가 동일한 array를 copy 없이 사용할 수 있고 서로 독립적인 stream에서 병렬 실행할 수 있다.

그러나 Apple Silicon CPU와 GPU는 같은 DRAM bandwidth를 공유한다.

따라서 discrete GPU FreeToken처럼

\[
B_{\text{CPU}} + B_{\text{GPU}}
\]

를 독립 bandwidth로 취급할 수 없다.

decode expert GEMV는 memory-bound일 가능성이 높다. CPU expert execution이 GPU bandwidth를 침범하면 hybrid가 오히려 느려질 수 있다.

따라서 MVP는 다음으로 고정한다.

- CPU는 routing metadata, cache, I/O, scheduling 담당
- expert 계산은 GPU 담당
- CPU expert compute는 benchmark로 이득이 확인된 후 별도 backend로 추가

이 결정은 이론적 금지가 아니라 **초기 복잡도와 shared-bandwidth 위험을 줄이는 설계 선택**이다.

---

## 14. memory budget 모델

총 Unified Memory를 \(M_{\text{phys}}\)라 하자.

실제 package가 사용할 수 있는 안전 working set을

\[
M_{\text{safe}}
<
M_{\text{phys}}
\]

로 둔다.

\[
M_{\text{safe}}
=
\min
(
M_{\text{user}},
M_{\text{recommended}}
)
-
M_{\text{OS-margin}}
\]

runtime budget은

\[
M_{\text{nonexpert}}
+
M_{\text{KV}}
+
M_{\text{expert}}
+
M_{\text{scratch}}
\le
M_{\text{safe}}
\]

를 만족해야 한다.

expert budget은

\[
M_{\text{expert}}
=
M_{\text{safe}}
-
M_{\text{nonexpert}}
-
M_{\text{KV reserve}}
-
M_{\text{scratch reserve}}
\]

로 계산한다.

MLX의 wired limit은 optional tuning으로만 사용한다. 기본 동작이 sudo 또는 sysctl 변경을 요구해서는 안 된다.

---

## 15. 모델 storage 전제

MVP는 Hugging Face safetensors를 source format으로 사용한다.

중요한 조건은 expert slice가 독립적으로 읽을 수 있어야 한다는 것이다.

다음 두 layout을 우선 지원한다.

### A. expert별 tensor

```text
model.layers.10.mlp.experts.0.up_proj.weight
model.layers.10.mlp.experts.1.up_proj.weight
...
```

### B. leading expert axis tensor

```text
model.layers.10.mlp.experts.gate_up_proj
shape = [E, ...]
```

Safetensors는 row-major data offset을 갖기 때문에 expert axis가 첫 차원이고 slice가 contiguous이면 exact byte range를 계산해 `pread`할 수 있다.

MVP의 핵심은 **모델 전체 tensor를 MLX로 load한 뒤 slice하는 것이 아니라, disk file에서 필요한 expert byte range만 직접 읽는 것**이다.

---

## 16. 온라인 quantization을 하지 않는 이유

miss마다 BF16 expert를 읽고 Q4로 quantize하면 I/O와 CPU 비용이 모두 늘어난다.

따라서 초기 runtime은 **이미 quantized된 expert representation**을 요구한다.

expert bundle은 packed weight뿐 아니라 scale 및 bias tensor를 같이 관리해야 한다.

한 번의 offline prepare 단계에서

- source model 검사
- expert tensor index 생성
- 필요하면 MLX-friendly quantized format 생성
- manifest 생성

을 수행한다.

---

## 17. 핵심 연구 가설

### H1 Decode locality

인접 token에서 동일 layer expert가 충분히 반복된다.

검증

\[
J_{t,l}
=
\frac{|S_{t,l}\cap S_{t-1,l}|}
{|S_{t,l}\cup S_{t-1,l}|}
\]

의 분포를 측정한다.

### H2 Small resident cache is useful

전체 expert pool보다 훨씬 작은 resident cache에서도 byte-weighted hit rate가 의미 있게 나온다.

### H3 Explicit residency beats OS swap

동일한 메모리 초과 조건에서 explicit expert cache + exact read가 MLX lazy load 또는 OS swap에 의존한 baseline보다 명확히 빠르다.

### H4 Prefill expert-major streaming is practical

긴 prompt에서 TTFT가 크더라도 naive per-token expert load보다 현저히 작아야 한다.

### H5 I/O overlap is measurable

GPU compute와 SSD read를 겹쳤을 때

\[
T_{\text{pipeline}}
<
T_{\text{read-only}}+T_{\text{compute-only}}
\]

가 유의하게 성립해야 한다.

이 가설 중 H1~H3가 실패하면 프로젝트의 핵심 타당성을 다시 검토한다.

---

## 18. 초기 성공 기준

다음은 연구 결과가 아니라 engineering target이다.

### Correctness Gate

- resident full-memory reference와 generated token sequence가 deterministic sampling 조건에서 동일하거나
- logits가 dtype에 맞는 tolerance 안에서 일치

### Oversubscription Gate

실제 물리 RAM보다 큰 모델로 가기 전에, 모델은 RAM에 들어가더라도 package의 `resident_budget`을 인위적으로 낮춰 전체 expert pool의 30%, 50%, 70% oversubscription을 재현한다.

### Performance Gate

최소한 다음을 만족해야 실제 over-memory model 단계로 진행한다.

- naive reload baseline 대비 decode throughput 2배 이상 또는
- OS swap/lazy oversubscription baseline 대비 현저한 개선
- cache hit 증가가 tok/s 증가와 일관된 상관을 보임
- profiler에서 full-model read가 아니라 selected expert byte만 read됨

### Release Candidate Target

적어도 한 configuration에서

- serialized model weight > configured safe resident budget by 20% 이상
- sustained generation >= 5 tok/s
- OOM 또는 swap-thrash 없이 256 output token 이상 생성
- OpenAI-compatible local API에서 연속 요청 성공

5 tok/s는 제품 보장이 아니라 최초 공개 가능성을 판단하기 위한 기준이다.

---

## 19. 실패 조건

다음 결과가 나오면 무리하게 최적화를 계속하지 않는다.

1. routing locality가 매우 낮아 resident cache가 거의 무효
2. expert bundle이 너무 커서 한 miss만으로 token latency budget을 대부분 소모
3. quantized expert slice를 독립적으로 읽을 수 없는 모델 format
4. MLX array materialization copy 비용이 SSD read 이득을 상쇄
5. prefill이 실제 사용 scenario에서 허용할 수 없을 정도로 느림
6. macOS memory pressure가 explicit cache에도 불구하고 전체 system을 swap-thrash로 밀어넣음
7. custom Metal 없이 Python dispatch overhead가 제거되지 않음

이 경우 다음 단계는 억지로 더 큰 모델을 띄우는 것이 아니라 storage format 또는 native backend를 재설계하는 것이다.

---

## 20. 이 프로젝트의 기술적 주장

이 프로젝트가 증명하려는 것은 다음 하나다.

> Apple Silicon에서 총 MoE weight 전체를 resident로 둘 수 없더라도, sparse routing과 temporal locality를 이용해 필요한 expert만 명시적으로 SSD에서 Unified Memory working set으로 materialize하고 유지하면, OS swap에 의존하는 일반 out-of-core inference보다 훨씬 실용적인 decode 성능을 낼 수 있다.

반대로 다음은 주장하지 않는다.

- 모든 MoE 모델이 빠르다
- RAM보다 몇 배 큰 모델도 항상 interactive하다
- SSD bandwidth만 높으면 해결된다
- FreeToken의 RTX 결과를 Mac에서 재현한다
- Dense model에도 같은 효과가 난다

---

## 21. 참고 자료

- FreeToken, arXiv 2608.16157  
  https://arxiv.org/abs/2608.16157
- FreeToken source  
  https://github.com/FlashML-org/FreeToken
- MLX Unified Memory  
  https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- MLX Custom Metal Kernels  
  https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- mlx-lm Qwen3 MoE implementation  
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_moe.py
- mlx-lm Switch layers  
  https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py
- Safetensors documentation  
  https://huggingface.co/docs/safetensors/index
- MLX mmap / over-memory discussion  
  https://github.com/ml-explore/mlx/discussions/615

