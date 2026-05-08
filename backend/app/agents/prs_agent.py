"""
agents/prs_agent.py

LangGraph node that handles PRS queries:
  - PNR status
"""

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from app.config import settings
from app.tools.pnr_tool import get_pnr_status
from app.agents.state import AgentState

llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

SYSTEM_PROMPT = """You are a helpful Indian Railways assistant specializing in PNR status and seat availability.

You have access to the following tools:
- get_pnr_status: Get live PNR status, passengers, and chart information.

Guidelines:
- Explain the PNR status clearly to the user.
- Include coach and berth information if available.
- Mention if the chart is prepared.
- If a tool returns an error (like Invalid PNR or Flushed), explain it politely.
"""

_react_agent = create_react_agent(
    model=llm,
    tools=[get_pnr_status],
)

async def prs_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    try:
        result = await _react_agent.ainvoke({"messages": full_messages})

        final_message = result["messages"][-1]
        response_text = final_message.content

        prs_output = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "name") and msg.name == "get_pnr_status":
                import json
                try:
                    prs_output = json.loads(msg.content)
                except Exception:
                    prs_output = {"raw": msg.content}
                break

        return {
            **state,
            "messages": result["messages"],
            "prs_output": prs_output,
            "response": response_text,
            "error": None,
            "error_code": None,
        }

    except Exception as e:
        return {
            **state,
            "response": None,
            "error": f"PRS agent error: {str(e)}",
            "error_code": "AGENT_ERROR",
        }
