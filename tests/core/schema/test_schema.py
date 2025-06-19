import pytest

from legisdefagent.core.schema import AgentInfo
from legisdefagent.core.schema.client import ChatMessage, Feedback, UserInput
from legisdefagent.core.schema.definition import DefinitionMetadata
from legisdefagent.core.schema.task_data import TaskData


class TestChatMessage:
    def test_pretty_repr_human(self):
        msg = ChatMessage(type="human", content="Hello")
        result = msg.pretty_repr()
        assert "Human Message" in result
        assert "Hello" in result

    def test_pretty_repr_ai(self):
        msg = ChatMessage(type="ai", content="Reply")
        result = msg.pretty_repr()
        assert "Ai Message" in result

    def test_defaults(self):
        msg = ChatMessage(type="human", content="test")
        assert msg.tool_calls == []
        assert msg.tool_call_id is None
        assert msg.response_metadata == {}


class TestAgentInfo:
    def test_creation(self):
        info = AgentInfo(key="TestAgent", description="A test agent")
        assert info.key == "TestAgent"
        assert info.description == "A test agent"


class TestTaskData:
    def test_completed_true(self):
        td = TaskData(state="complete", result="success")
        assert td.completed() is True

    def test_completed_false(self):
        td = TaskData(state="running")
        assert td.completed() is False

    def test_completed_with_error(self):
        td = TaskData(state="complete", result="error")
        assert td.completed_with_error() is True

    def test_completed_with_success_not_error(self):
        td = TaskData(state="complete", result="success")
        assert td.completed_with_error() is False

    def test_defaults(self):
        td = TaskData()
        assert td.name is None
        assert td.state is None
        assert td.data == {}


class TestUserInput:
    def test_defaults(self):
        ui = UserInput(message="What is X?")
        assert ui.model == "gpt-4o-mini"
        assert ui.thread_id is None

    def test_custom_values(self):
        ui = UserInput(message="test", model="custom-model", thread_id="abc-123")
        assert ui.model == "custom-model"
        assert ui.thread_id == "abc-123"


class TestDefinitionMetadata:
    def test_required_fields(self):
        dm = DefinitionMetadata(
            id=1,
            dataset="eu_legislation",
            document_id="doc_001.xml",
            definiendum_label="term",
            def_n="#def_1",
            frbr_work="/akn/eu/act/2020",
        )
        assert dm.id == 1
        assert dm.dataset == "eu_legislation"

    def test_missing_field_raises(self):
        with pytest.raises(Exception):
            DefinitionMetadata(id=1, dataset="test")


class TestFeedback:
    def test_creation(self):
        fb = Feedback(run_id="abc-123", key="stars", score=0.9)
        assert fb.run_id == "abc-123"
        assert fb.kwargs == {}
