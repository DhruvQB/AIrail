"""
agents/graph.py

Assembles the full LangGraph multi-agent graph.

Flow:
  START → supervisor → (railradar | prs | rag | guardrail | END for greetings)

The supervisor classifies intent, handles greetings inline, and routes
all other intents to the appropriate agent node.
"""

from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.supervisor import supervisor_node
from app.agents.railradar_agent import railradar_node
from app.agents.rag_agent import rag_agent_node
from app.agents.prs_agent import prs_node
from app.agents.guardrail_agent import guardrail_node


# ---------------------------------------------------------------------------
# Routing function — reads intent set by supervisor
# ---------------------------------------------------------------------------

def route_to_agent(state: AgentState) -> str:
    intent = state.get("intent", "rag")
    if intent == "railradar":
        return "railradar_agent"
    elif intent == "prs":
        return "prs_agent"
    elif intent == "guardrail":
        return "guardrail_agent"
    elif intent == "greeting":
        # Supervisor handled greetings inline and already set state["response"]
        return END
    else:
        return "rag_agent"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("railradar_agent", railradar_node)
    graph.add_node("prs_agent", prs_node)
    graph.add_node("rag_agent", rag_agent_node)
    graph.add_node("guardrail_agent", guardrail_node)

    # Edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "railradar_agent": "railradar_agent",
            "prs_agent": "prs_agent",
            "rag_agent": "rag_agent",
            "guardrail_agent": "guardrail_agent",
            END: END,
        }
    )
    graph.add_edge("railradar_agent", END)
    graph.add_edge("prs_agent", END)
    graph.add_edge("rag_agent", END)
    graph.add_edge("guardrail_agent", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Singleton — compiled once, reused across requests
# ---------------------------------------------------------------------------

agent_graph = build_graph()