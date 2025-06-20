from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from legisdefagent.core.agents import definitions_agent, definitions_agent_eval
from legisdefagent.core.schema import AgentInfo


DEFAULT_AGENT = "LegisDefAgent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "LegisDefAgent": Agent(
        description="A legal assistant for definition Retrieval and Generation", graph=definitions_agent
    ),
    "LegisDefAgentEval": Agent(
        description="A version of the legal definition assistant for evaluation purposes", graph=definitions_agent_eval
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()]
