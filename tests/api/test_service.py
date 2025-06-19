import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel.types import StateSnapshot

from legisdefagent.core.agents.registry import Agent
from legisdefagent.core.schema import ChatMessage
from legisdefagent.core.schema.client import ChatHistory, ServiceMetadata
from legisdefagent.core.schema.models import OpenAIModelName


def test_invoke(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."

    mock_agent.ainvoke.return_value = {"messages": [AIMessage(content=ANSWER)]}

    response = test_client.post("/invoke", json={"message": QUESTION})
    assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.model_validate(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER


@pytest.mark.asyncio
async def test_stream(test_client, mock_agent) -> None:
    """Test streaming tokens and messages."""
    QUESTION = "What is the weather in Tokyo?"
    TOKENS = ["The", " weather", " in", " Tokyo", " is", " sunny", "."]
    FINAL_ANSWER = "The weather in Tokyo is sunny."

    events = [
        {
            "event": "on_chat_model_stream",
            "tags": ["graph:step:chat_model"],
            "data": {"chunk": AIMessageChunk(content=token)},
        }
        for token in TOKENS
    ] + [
        {
            "event": "on_chain_end",
            "tags": ["graph:step:chat_model"],
            "data": {"output": {"messages": [AIMessage(content=FINAL_ANSWER)]}},
        }
    ]

    async def mock_astream_events(*args, **kwargs):
        for event in events:
            yield event

    mock_agent.astream_events = mock_astream_events

    # Make request with streaming
    with test_client.stream("POST", "/stream", json={"message": QUESTION, "stream_tokens": True}) as response:
        assert response.status_code == 200

        # Collect all SSE messages
        messages = []
        for line in response.iter_lines():
            if line and line.strip() != "data: [DONE]":  # Skip [DONE] message
                messages.append(json.loads(line.lstrip("data: ")))

        # Verify streamed tokens
        token_messages = [msg for msg in messages if msg["type"] == "token"]
        assert len(token_messages) == len(TOKENS)
        for i, msg in enumerate(token_messages):
            assert msg["content"] == TOKENS[i]

        # Verify final message
        final_messages = [msg for msg in messages if msg["type"] == "message"]
        assert len(final_messages) == 1
        assert final_messages[0]["content"]["content"] == FINAL_ANSWER
        assert final_messages[0]["content"]["type"] == "ai"


def test_history(test_client, mock_agent) -> None:
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    user_question = HumanMessage(content=QUESTION)
    agent_response = AIMessage(content=ANSWER)
    mock_agent.get_state.return_value = StateSnapshot(
        values={"messages": [user_question, agent_response]},
        next=(),
        config={},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )

    response = test_client.post("/history", json={"thread_id": "7bcc7cc1-99d7-4b1d-bdb5-e6f90ed44de6"})
    assert response.status_code == 200

    output = ChatHistory.model_validate(response.json())
    assert output.messages[0].type == "human"
    assert output.messages[0].content == QUESTION
    assert output.messages[1].type == "ai"
    assert output.messages[1].content == ANSWER


def test_info(test_client, mock_settings) -> None:
    """Test that /info returns the correct service metadata."""

    mock_graph = Mock(spec=CompiledStateGraph)

    base_agent = Agent(description="A base agent.", graph=mock_graph)
    mock_settings.auth_secret = None
    mock_settings.default_model = OpenAIModelName.GPT_4O
    mock_settings.available_models = {OpenAIModelName.GPT_4O, OpenAIModelName.GPT_4O_MINI}
    with patch.dict("legisdefagent.core.agents.registry.agents", {"base-agent": base_agent}, clear=True):
        response = test_client.get("/info")
        assert response.status_code == 200
        output = ServiceMetadata.model_validate(response.json())

    assert output.default_agent == "LegisDefAgent"
    assert len(output.agents) == 1
    assert output.agents[0].key == "base-agent"
    assert output.agents[0].description == "A base agent."

    assert output.default_model == OpenAIModelName.GPT_4O
    assert output.models == ["gpt-4o", "gpt-4o-mini"]
