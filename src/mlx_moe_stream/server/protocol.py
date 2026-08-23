"""OpenAI-shaped request validation and streaming-safe protocol transforms."""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal

from jsonschema import Draft202012Validator, ValidationError

Endpoint = Literal["completion", "chat_completion"]
Message = dict[str, Any]


class ApiRequestError(ValueError):
    """An OpenAI-shaped request that the local server cannot satisfy."""

    def __init__(
        self,
        message: str,
        *,
        status: HTTPStatus = HTTPStatus.BAD_REQUEST,
        parameter: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.parameter = parameter
        self.code = code


@dataclass(frozen=True)
class GenerationOptions:
    """Validated request controls that map to mlx-lm without approximation."""

    model_id: str
    max_tokens: int
    stream: bool
    temperature: float
    top_p: float
    stop: tuple[str, ...]
    seed: int | None
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict[int, float] | None
    tools: tuple[dict[str, Any], ...]
    required_tool_name: str | None
    tool_choice: str | None
    response_format: dict[str, Any] | None
    reasoning_effort: str
    include_usage: bool

    @property
    def template_tools(self) -> tuple[dict[str, Any], ...]:
        """The subset made available to the tokenizer chat template."""

        if self.tool_choice == "none":
            return ()
        if self.required_tool_name is None:
            return self.tools
        return tuple(
            tool for tool in self.tools if tool["function"]["name"] == self.required_tool_name
        )


def parse_generation_options(
    payload: dict[str, Any],
    *,
    endpoint: Endpoint,
    default_model_id: str,
    model_exists: Callable[[str], bool],
    max_completion_tokens: int,
) -> GenerationOptions:
    """Validate supported OpenAI fields and normalize their MLX equivalents."""

    model = payload.get("model", default_model_id)
    if not isinstance(model, str) or not model_exists(model):
        raise ApiRequestError(
            f"unknown local model {model!r}",
            status=HTTPStatus.NOT_FOUND,
            parameter="model",
            code="model_not_found",
        )
    stream = _boolean(payload, "stream", default=False)
    if payload.get("n", 1) != 1:
        raise ApiRequestError("only n=1 is supported", parameter="n")
    max_tokens = _max_tokens(payload, max_completion_tokens)
    temperature = _number(payload, "temperature", default=0.0, minimum=0.0, maximum=2.0)
    top_p = _number(payload, "top_p", default=1.0, minimum=0.0, maximum=1.0)
    presence_penalty = _number(payload, "presence_penalty", default=0.0, minimum=-2.0, maximum=2.0)
    frequency_penalty = _number(
        payload, "frequency_penalty", default=0.0, minimum=-2.0, maximum=2.0
    )
    stop = _parse_stop(payload.get("stop"))
    seed = _parse_seed(payload.get("seed"))
    logit_bias = _parse_logit_bias(payload.get("logit_bias"))
    tools, tool_choice, required_tool_name = _parse_tools(payload)
    response_format = _parse_response_format(payload.get("response_format"))
    reasoning_effort = _parse_reasoning_effort(payload.get("reasoning_effort", "medium"))
    include_usage = _parse_stream_options(payload.get("stream_options"))

    if endpoint == "completion":
        if tools or "tool_choice" in payload:
            raise ApiRequestError(
                "tools are supported only by /v1/chat/completions", parameter="tools"
            )
        if "reasoning_effort" in payload:
            raise ApiRequestError(
                "reasoning_effort is supported only by /v1/chat/completions",
                parameter="reasoning_effort",
            )
        if response_format is not None:
            raise ApiRequestError(
                "response_format is supported only by /v1/chat/completions",
                parameter="response_format",
            )
    if stream and response_format is not None:
        raise ApiRequestError(
            "streaming structured output is not enabled because schema validation occurs "
            "before a response is committed; use stream=false",
            parameter="response_format",
        )

    return GenerationOptions(
        model_id=model,
        max_tokens=max_tokens,
        stream=stream,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        seed=seed,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
        tools=tools,
        required_tool_name=required_tool_name,
        tool_choice=tool_choice,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
        include_usage=include_usage,
    )


def normalize_messages(messages: Any) -> list[Message]:
    """Normalize text and image chat content into MLX-VLM template messages.

    Images use the small common shape ``{"type": "image", "image": source}``
    expected by the Qwen and Gemma MLX-VLM processors.  Audio and video are
    intentionally rejected: M12 exposes image VLM, not an incomplete media API.
    """

    if not isinstance(messages, list) or not messages:
        raise ApiRequestError("'messages' must be a non-empty array", parameter="messages")
    normalized: list[Message] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ApiRequestError(f"messages[{index}] must be an object", parameter="messages")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ApiRequestError(f"messages[{index}].role is unsupported", parameter="messages")
        content = message.get("content")
        if content is None and role == "assistant":
            content = ""
        if isinstance(content, list):
            content = _normalize_content_parts(content, index=index, role=role)
        if not isinstance(content, (str, list)):
            raise ApiRequestError(
                f"messages[{index}].content must be a string or content-part array",
                parameter="messages",
            )
        item: Message = {"role": role, "content": content}
        if role == "assistant":
            reasoning_content = message.get("reasoning_content")
            if reasoning_content is not None:
                if not isinstance(reasoning_content, str):
                    raise ApiRequestError(
                        f"messages[{index}].reasoning_content must be a string",
                        parameter="messages",
                    )
                item["reasoning_content"] = reasoning_content
            if "tool_calls" in message:
                item["tool_calls"] = _normalize_prior_tool_calls(message["tool_calls"], index)
        elif role == "tool" and "tool_call_id" in message:
            if not isinstance(message["tool_call_id"], str):
                raise ApiRequestError(
                    f"messages[{index}].tool_call_id must be a string", parameter="messages"
                )
            item["tool_call_id"] = message["tool_call_id"]
        normalized.append(item)
    return normalized


def image_sources(messages: Iterable[Message]) -> list[str]:
    """Extract bounded image sources after :func:`normalize_messages`."""

    sources: list[str] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image":
                sources.append(part["image"])
    if len(sources) > 4:
        raise ApiRequestError("at most 4 images are supported per request", parameter="messages")
    return sources


def _normalize_content_parts(
    parts: list[Any], *, index: int, role: str
) -> str | list[dict[str, str]]:
    if role == "system":
        raise ApiRequestError(
            f"messages[{index}].content must be a string for system messages",
            parameter="messages",
        )
    if not parts:
        raise ApiRequestError(f"messages[{index}].content must not be empty", parameter="messages")
    normalized: list[dict[str, str]] = []
    has_image = False
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ApiRequestError(
                f"messages[{index}].content[{part_index}] must be an object", parameter="messages"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str):
                raise ApiRequestError(
                    f"messages[{index}].content[{part_index}].text must be a string",
                    parameter="messages",
                )
            normalized.append({"type": "text", "text": text})
            continue
        if part_type in {"image_url", "input_image", "image"}:
            if role != "user":
                raise ApiRequestError(
                    "image content is supported only in user messages", parameter="messages"
                )
            source = _image_source(part, part_type)
            normalized.append({"type": "image", "image": source})
            has_image = True
            continue
        if part_type in {"audio", "input_audio", "video", "input_video"}:
            raise ApiRequestError(
                "audio and video inputs are not supported; M12 supports images only",
                parameter="messages",
            )
        raise ApiRequestError(
            f"messages[{index}].content[{part_index}].type is unsupported",
            parameter="messages",
        )
    if not has_image:
        return "".join(part["text"] for part in normalized)
    return normalized


def _image_source(part: dict[str, Any], part_type: str) -> str:
    value = part.get("image") if part_type == "image" else part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value:
        raise ApiRequestError("image content requires a non-empty image URL", parameter="messages")
    return value


def add_response_format_instruction(
    messages: list[Message], response_format: dict[str, Any] | None
) -> list[Message]:
    """Ask the model for a JSON result before strict local validation."""

    if response_format is None:
        return messages
    response_type = response_format["type"]
    if response_type == "json_object":
        instruction = "Return exactly one valid JSON object. Do not use Markdown fences or prose."
    else:
        schema = response_format["json_schema"]["schema"]
        rendered_schema = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        instruction = (
            "Return exactly one valid JSON object matching this JSON Schema. "
            f"Do not use Markdown fences or prose. Schema: {rendered_schema}"
        )
    copied = [dict(message) for message in messages]
    if copied and copied[0]["role"] == "system":
        copied[0]["content"] = f"{copied[0]['content']}\n\n{instruction}".strip()
    else:
        copied.insert(0, {"role": "system", "content": instruction})
    return copied


def render_chat_prompt(
    messages: list[Message],
    tokenizer: Any,
    *,
    tools: Iterable[dict[str, Any]],
    reasoning_effort: str,
) -> tuple[str, bool]:
    """Render chat with only tokenizer-template features it actually declares."""

    template_tools = tuple(tools)
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        if template_tools:
            raise ApiRequestError(
                "selected tokenizer has no tool-aware chat template", parameter="tools"
            )
        if image_sources(messages):
            raise ApiRequestError(
                "selected model has no vision chat template", parameter="messages"
            )
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return rendered, False

    template = getattr(tokenizer, "chat_template", "")
    supports_tools = isinstance(template, str) and "tools" in template
    supports_thinking = isinstance(template, str) and "enable_thinking" in template
    if template_tools and not supports_tools:
        raise ApiRequestError(
            "selected tokenizer does not declare a tool-aware chat template", parameter="tools"
        )
    enable_thinking = reasoning_effort != "none" and supports_thinking
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if template_tools:
        kwargs["tools"] = list(template_tools)
    if supports_thinking:
        kwargs["enable_thinking"] = enable_thinking
    rendered = apply_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise ApiRequestError("tokenizer chat template did not return text")
    return rendered, enable_thinking


def validate_structured_output(text: str, response_format: dict[str, Any] | None) -> None:
    """Fail closed if a requested JSON response is syntactically or schema invalid."""

    if response_format is None:
        return
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ApiRequestError(
            "model output is not valid JSON for the requested response_format",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            parameter="response_format",
            code="invalid_structured_output",
        ) from error
    if not isinstance(value, dict):
        raise ApiRequestError(
            "response_format requires a JSON object",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            parameter="response_format",
            code="invalid_structured_output",
        )
    if response_format["type"] == "json_schema":
        schema = response_format["json_schema"]["schema"]
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError as error:
            raise ApiRequestError(
                f"model output does not match the requested JSON Schema: {error.message}",
                status=HTTPStatus.UNPROCESSABLE_ENTITY,
                parameter="response_format",
                code="invalid_structured_output",
            ) from error


def split_reasoning(text: str, *, initial_reasoning: bool) -> tuple[str, str]:
    """Return ``(reasoning_content, visible_content)`` from Qwen think tags."""

    parser = ThinkingStreamParser(initial_reasoning=initial_reasoning)
    reasoning: list[str] = []
    content: list[str] = []
    for kind, fragment in [*parser.feed(text), *parser.flush()]:
        (reasoning if kind == "reasoning_content" else content).append(fragment)
    return "".join(reasoning).strip(), "".join(content).strip()


class StopSequenceBuffer:
    """Hold possible stop prefixes so SSE never leaks a partial stop sequence."""

    def __init__(self, stops: tuple[str, ...]) -> None:
        self._stops = stops
        self._buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped or not text:
            return ""
        self._buffer += text
        found = [self._buffer.find(stop) for stop in self._stops if stop in self._buffer]
        if found:
            index = min(found)
            emitted = self._buffer[:index]
            self._buffer = ""
            self.stopped = True
            return emitted
        retained = _longest_stop_prefix(self._buffer, self._stops)
        if retained:
            emitted = self._buffer[: -len(retained)]
            self._buffer = retained
            return emitted
        emitted = self._buffer
        self._buffer = ""
        return emitted

    def flush(self) -> str:
        if self.stopped:
            return ""
        emitted = self._buffer
        self._buffer = ""
        return emitted


class ThinkingStreamParser:
    """Split a model's incremental text into OpenAI reasoning/content deltas."""

    def __init__(self, *, initial_reasoning: bool) -> None:
        self._in_reasoning = initial_reasoning
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buffer += text
        emitted: list[tuple[str, str]] = []
        while self._buffer:
            marker = "</think>" if self._in_reasoning else "<think>"
            index = self._buffer.find(marker)
            if index >= 0:
                if index:
                    emitted.append((self._field, self._buffer[:index]))
                self._buffer = self._buffer[index + len(marker) :]
                self._in_reasoning = not self._in_reasoning
                continue
            retained = _longest_marker_prefix(self._buffer, marker)
            if retained:
                prefix = self._buffer[: -len(retained)]
                if prefix:
                    emitted.append((self._field, prefix))
                self._buffer = retained
            else:
                emitted.append((self._field, self._buffer))
                self._buffer = ""
            break
        return emitted

    def flush(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        emitted = [(self._field, self._buffer)]
        self._buffer = ""
        return emitted

    @property
    def _field(self) -> str:
        return "reasoning_content" if self._in_reasoning else "content"


class ToolCallStreamParser:
    """Withhold Qwen XML tool calls until they can become valid OpenAI deltas."""

    _START = "<tool_call>"
    _END = "</tool_call>"

    def __init__(self, allowed_names: set[str]) -> None:
        self._allowed_names = allowed_names
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str, Any]]:
        self._buffer += text
        emitted: list[tuple[str, Any]] = []
        while self._buffer:
            start = self._buffer.find(self._START)
            if start < 0:
                retained = _longest_marker_prefix(self._buffer, self._START)
                content = self._buffer[: -len(retained)] if retained else self._buffer
                if content:
                    emitted.append(("content", content))
                self._buffer = retained
                break
            if start:
                emitted.append(("content", self._buffer[:start]))
                self._buffer = self._buffer[start:]
            end = self._buffer.find(self._END, len(self._START))
            if end < 0:
                break
            block_end = end + len(self._END)
            block = self._buffer[:block_end]
            _, calls = parse_tool_calls(block, self._allowed_names)
            emitted.append(("tool_calls", calls))
            self._buffer = self._buffer[block_end:]
        return emitted

    def flush(self) -> list[tuple[str, Any]]:
        if not self._buffer:
            return []
        buffered = self._buffer
        self._buffer = ""
        visible, calls = parse_tool_calls(buffered, self._allowed_names)
        emitted: list[tuple[str, Any]] = []
        if visible:
            emitted.append(("content", visible))
        if calls:
            emitted.append(("tool_calls", calls))
        return emitted


def parse_tool_calls(text: str, allowed_names: set[str]) -> tuple[str, list[dict[str, Any]]]:
    """Translate Qwen XML or common JSON tool syntax into OpenAI tool calls."""

    xml_matches = list(_TOOL_CALL_RE.finditer(text))
    if xml_matches:
        calls = [_xml_tool_call(match, allowed_names) for match in xml_matches]
        visible = _TOOL_CALL_RE.sub("", text).strip()
        return visible, calls

    stripped = text.strip()
    if not stripped:
        return "", []
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return text, []
    raw_calls: Any
    if isinstance(decoded, dict) and isinstance(decoded.get("tool_calls"), list):
        raw_calls = decoded["tool_calls"]
    elif isinstance(decoded, dict) and "name" in decoded and "arguments" in decoded:
        raw_calls = [decoded]
    else:
        return text, []
    calls = [_json_tool_call(raw_call, allowed_names) for raw_call in raw_calls]
    return "", calls


def enforce_tool_choice(options: GenerationOptions, calls: list[dict[str, Any]]) -> None:
    """Enforce choices that the tokenizer template cannot express directly."""

    if options.tool_choice == "required" and not calls:
        raise ApiRequestError(
            "tool_choice='required' but the model did not return a tool call",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            parameter="tool_choice",
            code="tool_choice_not_satisfied",
        )
    if options.required_tool_name is not None and any(
        call["function"]["name"] != options.required_tool_name for call in calls
    ):
        raise ApiRequestError(
            "model returned a tool outside the requested tool_choice",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            parameter="tool_choice",
            code="tool_choice_not_satisfied",
        )


def _max_tokens(payload: dict[str, Any], maximum: int) -> int:
    has_old = "max_tokens" in payload
    has_new = "max_completion_tokens" in payload
    if has_old and has_new:
        raise ApiRequestError(
            "supply only one of max_tokens and max_completion_tokens", parameter="max_tokens"
        )
    parameter = "max_completion_tokens" if has_new else "max_tokens"
    value = payload.get(parameter, maximum)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ApiRequestError(f"'{parameter}' must be a non-negative integer", parameter=parameter)
    if value > maximum:
        raise ApiRequestError(
            f"'{parameter}' exceeds the server limit ({maximum})", parameter=parameter
        )
    return value


def _number(
    payload: dict[str, Any], name: str, *, default: float, minimum: float, maximum: float
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ApiRequestError(f"'{name}' must be a finite number", parameter=name)
    number = float(value)
    if not minimum <= number <= maximum:
        raise ApiRequestError(
            f"'{name}' must be between {minimum:g} and {maximum:g}", parameter=name
        )
    return number


def _boolean(payload: dict[str, Any], name: str, *, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ApiRequestError(f"'{name}' must be a boolean", parameter=name)
    return value


def _parse_stop(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        raise ApiRequestError(
            "'stop' must be a string or an array of 1 to 4 strings", parameter="stop"
        )
    if any(not isinstance(item, str) or not item or len(item) > 256 for item in values):
        raise ApiRequestError(
            "each stop sequence must be a non-empty string of at most 256 characters",
            parameter="stop",
        )
    return tuple(values)


def _parse_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ApiRequestError("'seed' must be an unsigned 32-bit integer", parameter="seed")
    return value


def _parse_logit_bias(value: Any) -> dict[int, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ApiRequestError("'logit_bias' must be an object", parameter="logit_bias")
    normalized: dict[int, float] = {}
    for token_id, bias in value.items():
        try:
            parsed_id = int(token_id)
        except (TypeError, ValueError) as error:
            raise ApiRequestError(
                "logit_bias keys must be token IDs", parameter="logit_bias"
            ) from error
        if parsed_id < 0 or isinstance(bias, bool) or not isinstance(bias, (int, float)):
            raise ApiRequestError("logit_bias values must be numbers", parameter="logit_bias")
        parsed_bias = float(bias)
        if not math.isfinite(parsed_bias) or not -100 <= parsed_bias <= 100:
            raise ApiRequestError(
                "logit_bias values must be finite numbers between -100 and 100",
                parameter="logit_bias",
            )
        normalized[parsed_id] = parsed_bias
    return normalized


def _parse_tools(
    payload: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], str | None, str | None]:
    raw_tools = payload.get("tools")
    if raw_tools is None:
        if "tool_choice" in payload and payload["tool_choice"] not in (None, "none"):
            raise ApiRequestError("tool_choice requires tools", parameter="tool_choice")
        return (), "none" if payload.get("tool_choice") == "none" else None, None
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ApiRequestError("'tools' must be a non-empty array", parameter="tools")
    if len(raw_tools) > 128:
        raise ApiRequestError("at most 128 tools may be supplied", parameter="tools")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, dict) or raw_tool.get("type") != "function":
            raise ApiRequestError(f"tools[{index}] must be a function tool", parameter="tools")
        function = raw_tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ApiRequestError(
                f"tools[{index}].function.name must be a string", parameter="tools"
            )
        name = function["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) or name in names:
            raise ApiRequestError("tool names must be unique ASCII identifiers", parameter="tools")
        parameters = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ApiRequestError(
                f"tools[{index}].function.parameters must be an object", parameter="tools"
            )
        _validate_schema(parameters, parameter="tools")
        normalized_function: dict[str, Any] = {"name": name, "parameters": parameters}
        if "description" in function:
            if not isinstance(function["description"], str):
                raise ApiRequestError(
                    f"tools[{index}].function.description must be a string", parameter="tools"
                )
            normalized_function["description"] = function["description"]
        tools.append({"type": "function", "function": normalized_function})
        names.add(name)

    choice = payload.get("tool_choice", "auto")
    required_name: str | None = None
    if isinstance(choice, str) and choice in {"auto", "required", "none"}:
        pass
    elif isinstance(choice, dict):
        function = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function, dict):
            raise ApiRequestError("invalid function tool_choice", parameter="tool_choice")
        required_name = function.get("name")
        if not isinstance(required_name, str) or required_name not in names:
            raise ApiRequestError("tool_choice names an unavailable tool", parameter="tool_choice")
        choice = "function"
    else:
        raise ApiRequestError("invalid tool_choice", parameter="tool_choice")
    return tuple(tools), choice, required_name


def _parse_response_format(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ApiRequestError("'response_format' must be an object", parameter="response_format")
    response_type = value.get("type")
    if response_type == "text":
        return None
    if response_type == "json_object":
        return {"type": "json_object"}
    if response_type != "json_schema":
        raise ApiRequestError(
            "response_format.type must be text, json_object, or json_schema",
            parameter="response_format",
        )
    json_schema = value.get("json_schema")
    if not isinstance(json_schema, dict) or not isinstance(json_schema.get("name"), str):
        raise ApiRequestError(
            "json_schema response_format requires json_schema.name", parameter="response_format"
        )
    schema = json_schema.get("schema")
    if not isinstance(schema, dict):
        raise ApiRequestError(
            "json_schema response_format requires json_schema.schema", parameter="response_format"
        )
    _validate_schema(schema, parameter="response_format")
    return {"type": "json_schema", "json_schema": {"name": json_schema["name"], "schema": schema}}


def _parse_reasoning_effort(value: Any) -> str:
    if value not in {"none", "low", "medium", "high"}:
        raise ApiRequestError(
            "reasoning_effort must be one of none, low, medium, or high",
            parameter="reasoning_effort",
        )
    return value


def _parse_stream_options(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, dict):
        raise ApiRequestError("'stream_options' must be an object", parameter="stream_options")
    include_usage = value.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise ApiRequestError(
            "stream_options.include_usage must be a boolean", parameter="stream_options"
        )
    return include_usage


def _normalize_prior_tool_calls(value: Any, message_index: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ApiRequestError(
            f"messages[{message_index}].tool_calls must be a non-empty array", parameter="messages"
        )
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(value):
        if not isinstance(call, dict):
            raise ApiRequestError(
                "assistant tool_calls entries must be objects", parameter="messages"
            )
        function = call.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ApiRequestError("assistant tool call needs function.name", parameter="messages")
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ApiRequestError(
                    "assistant tool-call arguments must be JSON", parameter="messages"
                ) from error
        if not isinstance(arguments, dict):
            raise ApiRequestError(
                "assistant tool-call arguments must decode to an object", parameter="messages"
            )
        normalized.append(
            {
                "id": call.get("id", f"call_prior_{index}"),
                "type": "function",
                "function": {"name": function["name"], "arguments": arguments},
            }
        )
    return normalized


def _xml_tool_call(match: re.Match[str], allowed_names: set[str]) -> dict[str, Any]:
    name = match.group("name")
    body = match.group("body")
    arguments: dict[str, Any] = {}
    for parameter in _PARAMETER_RE.finditer(body):
        value = parameter.group("value").strip()
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        arguments[parameter.group("name")] = parsed_value
    return _new_tool_call(name, arguments, allowed_names)


def _json_tool_call(raw_call: Any, allowed_names: set[str]) -> dict[str, Any]:
    if not isinstance(raw_call, dict):
        raise ApiRequestError("model returned an invalid tool call", code="invalid_tool_call")
    function = raw_call.get("function", raw_call)
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise ApiRequestError("model returned an invalid tool call", code="invalid_tool_call")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise ApiRequestError(
                "model returned invalid tool-call JSON", code="invalid_tool_call"
            ) from error
    if not isinstance(arguments, dict):
        raise ApiRequestError(
            "model tool-call arguments must be an object", code="invalid_tool_call"
        )
    return _new_tool_call(function["name"], arguments, allowed_names, raw_call.get("id"))


def _new_tool_call(
    name: str, arguments: dict[str, Any], allowed_names: set[str], call_id: Any = None
) -> dict[str, Any]:
    if name not in allowed_names:
        raise ApiRequestError(
            f"model requested unavailable tool {name!r}",
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="invalid_tool_call",
        )
    if not isinstance(call_id, str):
        call_id = f"call_{uuid.uuid4().hex}"
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
        },
    }


def _validate_schema(schema: dict[str, Any], *, parameter: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        raise ApiRequestError(f"invalid JSON Schema: {error}", parameter=parameter) from error


def _longest_marker_prefix(text: str, marker: str) -> str:
    maximum = min(len(text), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if text.endswith(marker[:length]):
            return marker[:length]
    return ""


def _longest_stop_prefix(text: str, stops: tuple[str, ...]) -> str:
    candidates = [_longest_marker_prefix(text, stop) for stop in stops]
    return max(candidates, key=len, default="")


_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[A-Za-z0-9_-]+)>\s*"
    r"(?P<body>.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<name>[A-Za-z0-9_-]+)>\s*(?P<value>.*?)</parameter>", re.DOTALL
)
