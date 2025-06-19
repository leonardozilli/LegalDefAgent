import pytest
from langgraph.graph.state import CompiledStateGraph

from legisdefagent.core.agents.registry import get_agent, get_all_agent_info
from legisdefagent.core.schema import AgentInfo


class TestRegistry:
    def test_get_agent(self):
        agent = get_agent("LegisDefAgent")
        assert isinstance(agent, CompiledStateGraph)

        agent_eval = get_agent("LegisDefAgentEval")
        assert isinstance(agent_eval, CompiledStateGraph)

    def test_get_all_agent_info(self):
        info_list = get_all_agent_info()
        assert len(info_list) == 2

        keys = set(info.key for info in info_list)
        assert "LegisDefAgent" in keys
        assert "LegisDefAgentEval" in keys

        for info in info_list:
            assert isinstance(info, AgentInfo)
            assert info.description

    def test_get_agent_nonexistent(self):
        with pytest.raises(KeyError):
            get_agent("nonexistent-agent")
