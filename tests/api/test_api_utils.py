import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, ToolCall

from legisdefagent.api.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)


class TestConvertMessageContentToString:
    def test_string_passthrough(self):
        assert convert_message_content_to_string("hello") == "hello"

    def test_list_of_strings(self):
        assert convert_message_content_to_string(["hello", " world"]) == "hello world"

    def test_list_of_dicts(self):
        content = [{"type": "text", "text": "hello"}, {"type": "image", "url": "x.png"}]
        assert convert_message_content_to_string(content) == "hello"

    def test_mixed_list(self):
        content = ["prefix ", {"type": "text", "text": "suffix"}]
        assert convert_message_content_to_string(content) == "prefix suffix"


class TestLangchainToChatMessage:
    def test_human_message(self):
        msg = langchain_to_chat_message(HumanMessage(content="hi"))
        assert msg.type == "human"
        assert msg.content == "hi"

    def test_ai_message(self):
        msg = langchain_to_chat_message(AIMessage(content="response"))
        assert msg.type == "ai"
        assert msg.content == "response"

    def test_ai_message_with_tool_calls(self):
        ai = AIMessage(
            content="",
            tool_calls=[{"name": "search", "args": {"q": "test"}, "id": "tc_1"}],
        )
        msg = langchain_to_chat_message(ai)
        assert msg.type == "ai"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "search"

    def test_tool_message(self):
        msg = langchain_to_chat_message(ToolMessage(content="result", tool_call_id="tc_1"))
        assert msg.type == "tool"
        assert msg.tool_call_id == "tc_1"

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported message type"):
            langchain_to_chat_message(SystemMessage(content="sys"))

    def test_message_run_id_usage(self) -> None:
        run_id = "847c6285-8fc9-4560-a83f-4e6285809254"
        lc_message = AIMessage(content="Hello, world!")
        ai_message = langchain_to_chat_message(lc_message)
        ai_message.run_id = run_id
        assert ai_message.run_id == run_id

    def test_messages_tool_calls(self) -> None:
        tool_call = ToolCall(name="test_tool", args={"x": 1, "y": 2}, id="call_Jja7")
        lc_ai_message = AIMessage(content="", tool_calls=[tool_call])
        ai_message = langchain_to_chat_message(lc_ai_message)
        assert ai_message.tool_calls[0]["id"] == "call_Jja7"
        assert ai_message.tool_calls[0]["name"] == "test_tool"
        assert ai_message.tool_calls[0]["args"] == {"x": 1, "y": 2}


class TestRemoveToolCalls:
    def test_string_passthrough(self):
        assert remove_tool_calls("hello") == "hello"

    def test_filters_tool_use(self):
        content = [
            {"type": "text", "text": "hello"},
            {"type": "tool_use", "name": "search"},
        ]
        result = remove_tool_calls(content)
        assert len(result) == 1
        assert result[0]["type"] == "text"  # type: ignore

    def test_keeps_non_tool_use(self):
        content = ["plain string", {"type": "text", "text": "x"}]
        assert remove_tool_calls(content) == content
