# mlx-moe-stream 사용 가이드

[English README](README.md)

`mlx-moe-stream`은 Apple Silicon Mac에서 큰 MLX Mixture-of-Experts(MoE)
모델을 로컬로 구동하기 위한 서버입니다. 모든 전문가(expert) 가중치를
통합 메모리에 올려두지 않고, 현재 토큰과 레이어에서 라우터가 선택한 전문가만
SSD에서 정확한 범위로 읽어 MLX에 올립니다.

즉, **모델 전체가 SSD에서 실행되는 것은 아닙니다.** 라우터, 어텐션,
공유/밀집 가중치, 활성 전문가, KV cache는 통합 메모리를 사용합니다. 이미지
모드에서는 vision tower도 통합 메모리에 남습니다. SSD에는 주로 선택되지 않은
MoE 전문가 가중치가 머뭅니다.

> 이 프로젝트는 사전 릴리스 상태입니다. 인증이 없는 로컬 단일 사용자 서버로,
> 기본적으로 `127.0.0.1`에만 바인딩되며 메모리 예측성을 위해 한 번에 하나의
> 생성만 실행합니다.

## 지원 기능

| 목적 | 지원 방식 |
| --- | --- |
| 텍스트 completion / chat | `/v1/completions`, `/v1/chat/completions` |
| 토큰 스트리밍 | OpenAI 형식 SSE (`stream: true`) |
| thinking / reasoning | 모델이 출력하면 `reasoning_content`로 반환 |
| 함수 호출 | 호환되는 chat template에서 `tools`, `tool_choice` 지원 |
| 구조화된 JSON | 비스트리밍 `json_object`, `json_schema` |
| 이미지 채팅 | `--vision` 사용 시 Qwen3.6, Gemma 4 |
| 여러 모델 등록 | `--model 모델ID=MANIFEST` 반복; 메모리에는 한 모델만 활성화 |

## 요구 사항

- Apple Silicon Mac 및 macOS
- Python 3.10 이상
- 체크포인트 저장 및 추론 중 SSD 읽기를 위한 충분한 디스크 공간
- 지원하는 MLX safetensors MoE 체크포인트

대표 예시는 `mlx-community/Qwen3.6-35B-A3B-8bit`입니다. 테스트한 모델은
다음과 같습니다.

- `mlx-community/Qwen3.6-35B-A3B-8bit`
- `mlx-community/Qwen3-30B-A3B-4bit`
- `mlx-community/gemma-4-26b-a4b-it-8bit`

## M4 MacBook Pro 24GB 검증

24GB 통합 메모리를 갖춘 Apple M4 MacBook Pro에서 아래 모델을 로컬 서빙하고
추론하는 end-to-end 검증을 수행했습니다.

- `mlx-community/Qwen3.6-35B-A3B-8bit`
- `mlx-community/gemma-4-26b-a4b-it-8bit` (Gemma 4 8-bit의 정확한 저장소 ID)

두 모델은 큰 MoE 체크포인트입니다. SSD streaming은 선택되지 않은 expert를 통합
메모리에 계속 유지하지 않지만, model shell, 현재 활성 expert, KV cache는 여전히
통합 메모리를 사용합니다. 메모리 여유가 작다면 아래 장문 컨텍스트 설정처럼
prefill 속도와 피크 메모리를 맞바꾸세요.

### 256K 컨텍스트 윈도우 실측

같은 M4 환경에서 Qwen3.6으로 실제 OpenAI 호환
`/v1/chat/completions` 요청을 완료했습니다. **262,143개 prompt 토큰과 1개 생성
토큰**, 즉 **총 262,144개 토큰**을 처리했습니다. `--resident-budget off`,
`--kv-cache 4bit`, `--prefill-step-size 1024` 구성으로 2시간 3분 3초가 걸렸고,
MLX peak allocation은 12.86GiB였습니다. 이는 256K급 **컨텍스트 윈도우** 검증이지,
256K개의 출력 토큰을 대화형 속도로 생성한다는 뜻은 아닙니다. SSD-streamed MoE의
전체 윈도우 prefill은 의도적으로 느린 스트레스 테스트입니다.

### 지속 고부하 사용과 하드웨어 관리 안내

긴 생성은 CPU/GPU와 SSD I/O를 수 시간 동안 높은 수준으로 사용할 수 있습니다.
이는 Mac의 열 보호 기능을 우회하는 작업이 아닙니다. Apple 노트북은 내부 온도를
감지하고 중요 부품을 자동으로 냉각합니다. 본체가 따뜻해지고 fan이 있는 모델은
fan 속도가 올라갈 수 있습니다. 10–35°C 환경에서 통풍이 되는 단단하고 평평한
표면에 두고 사용하세요. 침구·베개 위나 가방 안에서는 실행하지 마세요.
[Apple의 온도 관리 안내](https://support.apple.com/en-us/102336)도 참고하세요.

장기적으로 신경 쓸 대상은 한 번의 추론보다 **배터리**입니다. 리튬 이온 배터리의
노화는 온도 이력과 충전 패턴의 영향을 받습니다. 수 시간씩 반복 서빙한다면 신뢰할
수 있는 전원 어댑터를 사용하고, **최적화된 배터리 충전** 및 적절한 **충전 제한**을
켜 두세요. Mac이 비정상적으로 뜨겁거나 온도·충전 경고를 표시하면 작업을
중지하세요. [Apple Silicon 배터리 상태 관리](https://support.apple.com/en-mide/102589)와
[충전 제한 안내](https://support.apple.com/en-au/102338)를 참고하세요.

SSD streaming 추론은 주로 **읽기** I/O이므로, 일반적으로 SSD 수명에 큰 영향을
주는 작업은 아닙니다. 다만 모델 다운로드, 반복 변환·양자화, 큰 trace, 애플리케이션
로그는 쓰기 작업입니다. 이러한 출력은 크기를 제한하고, SSD에 충분한 여유 공간을
남겨 두세요.

## 설치

PyPI 정식 배포 전까지는 공개 GitHub `main`에서 바로 설치합니다. Homebrew/system
Python에 설치하지 말고, Python 3.10+ 가상환경에서 아래 명령을 사용하세요.

```bash
python3.12 -m venv ~/.venvs/mlx-moe-stream
source ~/.venvs/mlx-moe-stream/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade \
  "mlx-moe-stream[vlm] @ git+https://github.com/IDAH-BITBOX/serve-mlx.git@main"
```

`[vlm]`은 이미지 채팅 지원을 설치합니다. 텍스트 전용이면 가상환경을 활성화한 뒤
아래 명령을 사용합니다.

```bash
python3 -m pip install --upgrade \
  "mlx-moe-stream @ git+https://github.com/IDAH-BITBOX/serve-mlx.git@main"
```

로컬에서 개발하려면 저장소를 clone한 뒤 가상환경에 설치합니다.

```bash
git clone https://github.com/IDAH-BITBOX/serve-mlx.git
cd serve-mlx
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

로컬 checkout에서 이미지 입력도 사용할 경우에는 VLM 의존성을 함께 설치합니다.

```bash
python -m pip install -e ".[vlm]"
```

설치 확인:

```bash
mlx-moe-stream --help
```

## 가장 빠른 시작: 텍스트 모델

### 1. 모델 manifest 만들기

`prepare`는 각 expert가 safetensors 파일의 어느 바이트 범위에 있는지 기록한
작은 manifest를 만듭니다. 가중치를 복사하지 않으며, 모델당 한 번만 실행하면
됩니다.

```bash
mlx-moe-stream prepare \
  --model mlx-community/Qwen3.6-35B-A3B-8bit \
  --output prepared-qwen3.6-35b
```

완료되면 `prepared-qwen3.6-35b/manifest.json`이 생성됩니다.

### 2. 로컬 서버 실행

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --model-id qwen3.6-35b-local \
  --resident-budget auto \
  --kv-cache auto
```

서버 주소는 `http://127.0.0.1:8000`이며 `Ctrl-C`를 누를 때까지 계속 실행됩니다.
요청 하나가 끝나도 모델을 내리거나 서버를 종료하지 않습니다. 첫 요청에서는
non-expert shell을 올려야 하므로 다음 요청보다 오래 걸릴 수 있습니다.

### 3. 채팅 요청 보내기

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-local",
    "messages": [{"role": "user", "content": "sparse MoE routing을 두 문장으로 설명해줘."}],
    "max_tokens": 128,
    "temperature": 0.2
  }'
```

상태와 메모리 판단 결과를 확인하려면 다음 엔드포인트를 사용합니다.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics
```

## OpenAI Python SDK에서 사용하기

OpenAI 호환 클라이언트의 base URL만 로컬 서버로 지정하면 됩니다. 실제 API key는
필요 없지만, SDK는 비어 있지 않은 값 하나를 요구합니다.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "MoE 모델의 용도 세 가지를 알려줘."}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### 스트리밍

```python
stream = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "SSD에 관한 짧은 하이쿠를 써줘."}],
    max_tokens=128,
    stream=True,
    stream_options={"include_usage": True},
)
for event in stream:
    text = event.choices[0].delta.content if event.choices else None
    if text:
        print(text, end="", flush=True)
```

## 이미지 채팅: Qwen3.6 / Gemma 4

이미지 입력을 사용하려면 `[vlm]`을 설치하고 서버를 `--vision`으로 시작해야 합니다.

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

사용자 메시지에서 **이미지 바이트를 직접 반환하는 URL**(`image/jpeg`,
`image/png` 등), `data:` URL, 또는 로컬 파일 경로를 전송할 수 있습니다. 요청당
최대 네 장입니다. Google 공유 링크나 HTML 뷰어 페이지는 이미지 URL이 아니므로,
원본/다운로드 이미지 URL을 사용하세요. 오디오와 비디오는 지원하지 않습니다.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-local",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "input_image", "image_url": "https://example.com/photo.jpg"},
        {"type": "text", "text": "이 이미지를 간단히 설명해줘."}
      ]
    }],
    "max_tokens": 128
  }'
```

Gemma 4를 사용하려면 prepare 단계의 모델 ID만 다음으로 바꾸면 됩니다.

```text
mlx-community/gemma-4-26b-a4b-it-8bit
```

## KV cache 용량과 정밀도

컨텍스트가 길어질수록 KV cache가 많은 통합 메모리를 사용합니다. `serve`와
`generate`에서 `--kv-cache` 옵션으로 정밀도를 선택할 수 있습니다.

```bash
--kv-cache auto   # 기본값: 실제 메모리 여유를 보고 자동 선택
--kv-cache bf16   # MLX의 native / 비양자화 KV cache
--kv-cache 8bit   # 양자화 KV cache, 메모리 절약
--kv-cache 4bit   # 가장 작은 KV cache, 큰 모델·긴 컨텍스트용
```

`auto`는 shell을 실제로 로드하고 크기를 측정한 다음 아래 순서로 선택합니다.

1. shell 이후 통합 메모리 여유가 충분하면 native BF16/FP KV cache를 선택합니다.
2. BF16 추정치가 KV 허용량을 넘으면 8-bit KV cache를 선택합니다.
3. 8-bit도 넘으면(oversized model 또는 최대 컨텍스트 등) 4-bit KV cache를 선택합니다.

결정된 모드, 예상 KV bytes, 예약 용량은 서버의 `/metrics`에서 볼 수 있고,
`generate`에서는 로그 및 `--memory-summary` JSON에 기록됩니다. 큰 KV cache가
필요하면 expert cache에 남는 자동 예산은 그만큼 줄어듭니다.

`generate`에서는 planner가 예약할 최대 전체 컨텍스트(프롬프트 + 완성 토큰)를
`--kv-max-context`로 알려주세요.

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --prompt "긴 문서를 요약해줘..." \
  --max-tokens 512 \
  --kv-cache auto \
  --kv-max-context 8192 \
  --resident-budget auto
```

`serve`에서는 `--max-prompt-tokens + --max-tokens` 값으로 KV cache 예약량을
계산합니다. 따라서 이 두 제한은 실제 사용량에 맞게 지정해야 합니다.
`--kv-reserve`는 최소 안전 예약량일 뿐, KV 정밀도를 정하는 옵션이 아닙니다.

큰 컨텍스트의 prefill에서 Metal out-of-memory가 발생한다면
`--prefill-step-size`를 낮추세요. 컨텍스트 한도와 별개로 한 번에 처리할 prompt
토큰 수를 제한합니다. 값이 작을수록 일시적인 MoE 활성화 메모리는 줄지만, prefill은
더 오래 걸립니다.

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

이는 **컨텍스트 윈도우** 설정입니다. 256K개 출력 토큰을 즉시 생성한다는 뜻이
아닙니다. SSD에서 expert를 스트리밍하는 MoE 모델의 전체 윈도우 prefill은 오래
걸릴 수 있습니다.

양자화 KV cache는 메모리를 줄이지만 생성 품질에 영향을 줄 수 있습니다. 짧은
컨텍스트에서 품질이 가장 중요하면 `bf16`을, 저사양 Mac에서 큰 모델이나 긴
컨텍스트를 다룬다면 `auto`, `8bit`, `4bit`을 권장합니다.

### 통합 메모리 여유 확보

이제 기본값인 `--memory-safety-margin auto`는 물리 통합 메모리의 25%를 expert
cache에 배정하지 않고 남겨 둡니다. 8GB Mac은 2GiB, 16GB Mac은 4GiB, 24GB Mac은
6GiB를 확보하며 최대 8GiB로 제한합니다. 실제 계산 결과는 `/metrics`의
`memory_budget`에서 확인할 수 있습니다.

더 큰 여유가 필요하면 resident cache를 명시적으로 제한하고 예약량을 늘리세요.

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

안전 여유는 resident expert cache에서 제외되지만, 모든 모델이 어느 Mac에서나
실행된다는 보장은 아닙니다. 특히 35B 모델은 non-expert shell만으로 8GB·16GB
Mac의 한계를 넘을 수 있습니다. 이 경우 M7은 swap으로 시스템 전체를 멈추게 하는
대신 expert-cache 계획을 거부합니다.

지속적으로 점유하는 메모리를 가장 작게 하려면 `--resident-budget off`로 resident
expert cache를 완전히 끌 수 있습니다. expert는 여전히 SSD에서 읽고 활성 레이어에
맞춰 materialize되지만, 이후에는 유지하지 않습니다.

```bash
mlx-moe-stream serve --manifest prepared-qwen3.6-35b/manifest.json \
  --resident-budget off --kv-cache 4bit --max-prompt-tokens 1024 --max-tokens 64
```

## 여러 모델 서빙하기

여러 manifest를 ID로 등록할 수 있습니다. `/v1/models`에 모두 노출되지만, 통합
메모리에는 현재 요청된 엔진 하나만 로드됩니다. 모델을 바꾸면 기존 엔진을 내리고
선택한 모델을 올립니다.

```bash
mlx-moe-stream serve \
  --model qwen=prepared-qwen3.6-35b/manifest.json \
  --model gemma=prepared-gemma4/manifest.json \
  --model-id qwen \
  --vision \
  --resident-budget auto \
  --kv-cache auto
```

API 요청의 `model`에 `qwen` 또는 `gemma`를 넣습니다. 한 서버에 등록된 모든
모델은 메모리, 최대 컨텍스트, KV cache, `--vision` 정책을 공유합니다. 모델마다
다른 정책이 필요하다면 서버 프로세스를 분리하세요.

## Thinking, tool calling, JSON

호환되는 tokenizer chat template을 가진 모델이라면, chat completion 요청의
`tools`, `tool_choice`, thinking 설정을 모델에 전달합니다. 모델의 호환되는 출력은
OpenAI 형태의 `reasoning_content`, `tool_calls`로 변환됩니다.

```python
response = client.chat.completions.create(
    model="qwen3.6-35b-local",
    messages=[{"role": "user", "content": "서울 날씨를 알려줘."}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "현재 날씨를 조회한다.",
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

`response_format`은 비스트리밍 chat 요청에서 `json_object`와 `json_schema`를
지원합니다. 응답을 반환하기 전에 JSON을 검증하므로, 아직 완료되지 않은 출력을
안전하게 검증할 수 없는 스트리밍 structured output은 의도적으로 허용하지 않습니다.

## 단발 생성과 라우팅 분석

서버를 띄우지 않고 텍스트를 한 번 생성하려면 `generate`를 사용합니다.

```bash
mlx-moe-stream generate \
  --manifest prepared-qwen3.6-35b/manifest.json \
  --prompt "sparse MoE routing을 설명해줘." \
  --max-tokens 64 \
  --resident-budget auto \
  --kv-cache auto
```

Qwen3-MoE의 라우터 선택을 확인하려면 `trace`, expert-cache 지역성을 보려면
`simulate`를 사용합니다. 현재 `trace`는 Qwen3-MoE 계열 전용이므로, 이 별도
예시에서는 Qwen3-30B를 사용합니다.

```bash
mlx-moe-stream trace \
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --prompt "Explain sparse MoE routing." \
  --max-tokens 64 \
  --output routes.jsonl \
  --summary routes-summary.json

mlx-moe-stream simulate --trace routes.jsonl
```

## 권장 튜닝 순서

1. 우선 `--resident-budget auto --kv-cache auto`로 시작합니다.
2. `--max-prompt-tokens`, `--max-tokens`은 실제 사용량에 가깝게 설정합니다.
3. 메모리 압박 또는 swap이 보이면 컨텍스트를 먼저 줄이고, 그 다음
   `--kv-cache 8bit` 또는 `--kv-cache 4bit`을 선택합니다.
4. 메모리는 여유 있지만 느리다면 resident expert 예산을 적절히 늘려 SSD 읽기를
   줄입니다. `/metrics` 또는 시작 로그에 나온 안전 예산을 넘기지 마세요.
5. 이 서버는 인증 기능이 없으므로 loopback에 두거나, 외부 노출이 필요하면 인증된
   reverse proxy 뒤에 배치하세요.

## 개발

테스트 도구까지 설치하려면 `python -m pip install -e ".[dev,vlm]"`를 실행한 뒤
다음을 사용합니다.

```bash
ruff check src tests
pytest
```

라이선스는 Apache-2.0입니다.
