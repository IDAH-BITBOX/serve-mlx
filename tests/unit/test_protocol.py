from __future__ import annotations

import pytest

from mlx_moe_stream.server.protocol import (
    ThinkingStreamParser,
    ToolCallStreamParser,
    normalize_messages,
    parse_generation_options,
    parse_tool_calls,
    split_reasoning,
)


def test_gemma_channel_reasoning_separates_from_visible_text_across_fragments():
    parser = ThinkingStreamParser(initial_reasoning=True)

    events = [
        *parser.feed("<|channel>tho"),
        *parser.feed("ught\ncheck the image"),
        *parser.feed("<|channel>fi"),
        *parser.feed("nal\nIt is blue."),
        *parser.flush(),
    ]

    assert events == [
        ("reasoning_content", "check the image"),
        ("content", "It is blue."),
    ]
    assert split_reasoning(
        "<|channel>thought\nstep<|channel>final\nanswer", initial_reasoning=True
    ) == ("step", "answer")


def test_gemma_native_tool_call_becomes_openai_function_call():
    text = (
        'I\'ll look it up. <|tool_call>call:get_weather{city:<|"|>Seoul<|"|>,'
        'options:{units:<|"|>metric<|"|>,days:2}}<tool_call|>'
    )

    visible, calls = parse_tool_calls(text, {"get_weather"})

    assert visible == "I'll look it up."
    assert calls[0]["type"] == "function"
    assert calls[0]["function"] == {
        "name": "get_weather",
        "arguments": '{"city":"Seoul","options":{"units":"metric","days":2}}',
    }


def test_gemma_native_tool_call_is_withheld_until_its_end_marker():
    parser = ToolCallStreamParser({"get_weather"})

    first = parser.feed('before <|tool_call>call:get_weather{city:<|"|>Seoul')
    second = parser.feed('<|"|>}<tool_call|> after')

    assert first == [("content", "before ")]
    assert second[0][0] == "tool_calls"
    assert second[0][1][0]["function"]["name"] == "get_weather"
    assert second[1:] == [("content", " after")]


def test_hermes_developer_instruction_normalizes_to_system_message():
    assert normalize_messages([{"role": "developer", "content": "Follow this policy."}]) == [
        {"role": "system", "content": "Follow this policy."}
    ]


@pytest.mark.parametrize(
    "effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
)
def test_hermes_reasoning_effort_range_is_accepted(effort: str):
    options = parse_generation_options(
        {"model": "qwen", "reasoning_effort": effort},
        endpoint="chat_completion",
        default_model_id="qwen",
        model_exists=lambda model: model == "qwen",
        max_completion_tokens=256,
    )

    assert options.reasoning_effort == effort
