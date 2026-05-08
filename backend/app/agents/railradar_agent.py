# Live status + train search agent
"""
agents/railradar_agent.py

LangGraph node that handles all RailRadar-related queries:
  - Live train status
  - Trains between stations
  - Train schedule
  - Live station board

The agent uses a ReAct loop — it reasons about which tool to call,
calls it, reads the result, and produces a natural language response.
"""

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.tools.railradar_tool import railradar_tools
from app.agents.state import AgentState


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful Indian Railways assistant specializing in live train information.

You have access to the following tools:
- get_live_train_status: Get real-time position, delay, and status of a train
- search_trains_between: Find all trains running between two stations
- get_train_schedule: Get the full stop-by-stop schedule of a train
- get_live_station_board: Get live arrivals and departures at a station
- lookup_station_by_name: Find station codes by searching station/city names (e.g., 'Surat' -> 'ST')

Guidelines:
- Always use station codes in UPPERCASE (e.g. ST for Surat, BCT for Mumbai Central, NDLS for New Delhi).
- Train numbers are strictly 5 digits.
- IMPORTANT: If a user provides a station name or city name instead of a code, you MUST use the `lookup_station_by_name` tool to find the correct station code before calling other tools.
- CRITICAL: Use ONLY the station codes returned by the `lookup_station_by_name` tool. Do not guess codes or use your internal knowledge. Do not substitute a major station (like Ahmedabad - ADI) for a smaller one (like Ambli Road - ABD) if the user specifically asked for the smaller one.
- If a user gives a train name but no train number, and you are not absolutely certain of the 5-digit number, ask the user for the 5-digit train number.
- When formatting your final response based on tool outputs, always display stations as 'Full Name - CODE' (e.g., 'Chandlodiya - CLDY') and trains as 'Train Name - Train Number'.
- If a tool returns an error, explain it clearly and helpfully to the user.
- Keep responses concise but informative.
- Always mention delay information when available.
- Format times as HH:MM (24-hour).
"""

# ---------------------------------------------------------------------------
# Internal react agent
# ---------------------------------------------------------------------------

_react_agent = create_react_agent(
    model=llm,
    tools=railradar_tools,
)

# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

async def railradar_node(state: AgentState) -> AgentState:
    """
    LangGraph node for RailRadar queries.
    Receives the full AgentState, runs the ReAct agent, 
    and writes the result back into state.
    """
    messages = state["messages"]

    # Prepend system prompt to the conversation
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    try:
        result = await _react_agent.ainvoke({"messages": full_messages})

        # Extract the final assistant text response
        final_message = result["messages"][-1]
        response_text = final_message.content

        # Extract tool output if available (last tool result before final message)
        railradar_output = None
        for msg in reversed(result["messages"]):
            if hasattr(msg, "name") and msg.name in [t.name for t in railradar_tools]:
                import json
                try:
                    railradar_output = json.loads(msg.content)
                except Exception:
                    railradar_output = {"raw": msg.content}
                break

        return {
            **state,
            "messages": result["messages"],
            "railradar_output": railradar_output,
            "response": response_text,
            "error": None,
            "error_code": None,
        }

    except Exception as e:
        error_msg = f"RailRadar agent encountered an error: {str(e)}"
        return {
            **state,
            "response": None,
            "error": error_msg,
            "error_code": "AGENT_ERROR",
        }