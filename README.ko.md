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

테스트한 모델 예시는 다음과 같습니다.

- `mlx-community/Qwen3-30B-A3B-4bit`
- `mlx-community/Qwen3.6-35B-A3B-8bit`
- `mlx-community/gemma-4-26b-a4b-it-8bit`

## 설치

저장소를 clone한 뒤 가상환경에 설치합니다.

```bash
git clone https://github.com/IDAH-BITBOX/serve-mlx.git
cd serve-mlx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

이미지 입력도 사용할 경우에는 VLM 의존성을 함께 설치합니다.

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
  --model mlx-community/Qwen3-30B-A3B-4bit \
  --output prepared-qwen3
```

완료되면 `prepared-qwen3/manifest.json`이 생성됩니다.

### 2. 로컬 서버 실행

```bash
mlx-moe-stream serve \
  --manifest prepared-qwen3/manifest.json \
  --model-id qwen3-local \
  --resident-budget auto \
  --kv-cache auto
```

서버 주소는 `http://127.0.0.1:8000`입니다. 첫 요청에서는 non-expert shell을
올려야 하므로 다음 요청보다 오래 걸릴 수 있습니다.

### 3. 채팅 요청 보내기

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-local",
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
    model="qwen3-local",
    messages=[{"role": "user", "content": "MoE 모델의 용도 세 가지를 알려줘."}],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### 스트리밍

```python
stream = client.chat.completions.create(
    model="qwen3-local",
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
  --output prepared-qwen3.6

mlx-moe-stream serve \
  --manifest prepared-qwen3.6/manifest.json \
  --model-id qwen3.6-local \
  --vision \
  --kv-cache auto
```

사용자 메시지에서 이미지 URL, `data:` URL, 또는 로컬 파일 경로를 전송할 수
있습니다. 요청당 최대 네 장이며 오디오와 비디오는 지원하지 않습니다.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-local",
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
  --manifest prepared-qwen3/manifest.json \
  --prompt "긴 문서를 요약해줘..." \
  --max-tokens 512 \
  --kv-cache auto \
  --kv-max-context 8192 \
  --resident-budget auto
```

`serve`에서는 `--max-prompt-tokens + --max-tokens` 값으로 KV cache 예약량을
계산합니다. 따라서 이 두 제한은 실제 사용량에 맞게 지정해야 합니다.
`--kv-reserve`는 최소 안전 예약량일 뿐, KV 정밀도를 정하는 옵션이 아닙니다.

양자화 KV cache는 메모리를 줄이지만 생성 품질에 영향을 줄 수 있습니다. 짧은
컨텍스트에서 품질이 가장 중요하면 `bf16`을, 저사양 Mac에서 큰 모델이나 긴
컨텍스트를 다룬다면 `auto`, `8bit`, `4bit`을 권장합니다.

## 여러 모델 서빙하기

여러 manifest를 ID로 등록할 수 있습니다. `/v1/models`에 모두 노출되지만, 통합
메모리에는 현재 요청된 엔진 하나만 로드됩니다. 모델을 바꾸면 기존 엔진을 내리고
선택한 모델을 올립니다.

```bash
mlx-moe-stream serve \
  --model qwen=prepared-qwen3.6/manifest.json \
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
    model="qwen3-local",
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
  --manifest prepared-qwen3/manifest.json \
  --prompt "sparse MoE routing을 설명해줘." \
  --max-tokens 64 \
  --resident-budget auto \
  --kv-cache auto
```

Qwen3-MoE의 라우터 선택을 확인하려면 `trace`, expert-cache 지역성을 보려면
`simulate`를 사용합니다.

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
