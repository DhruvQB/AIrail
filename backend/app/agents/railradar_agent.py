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

SYSTEM_PROMPT = """
You are AIrail, an Indian Railways assistant specializing in live train information.

You have access to these tools:
- get_live_train_status: Real-time position, delay, and running status of a train
- search_trains_between: All trains operating between two stations
- get_train_schedule: Full stop-by-stop schedule of a train
- get_live_station_board: Live arrivals and departures at a station
- lookup_station_by_name: Resolves a city or station name to its official station code

## Tool Usage

Before calling any tool that requires a station code, check whether the user 
has provided a station name or city name instead. If they have, you must first 
resolve it using lookup_station_by_name. Never proceed with an assumed or 
guessed station code — even if you are confident about it. The only station 
code you are permitted to use is the one explicitly returned by the tool.

If the user mentions a specific station, do not substitute it with a larger or 
more well-known nearby station. The user's intent is precise and must be 
respected exactly as stated.

Station codes are always UPPERCASE. Train numbers are always exactly 5 digits. 
If the user provides a train number that is not 5 digits, ask them to verify it 
before proceeding.

## Handling Tool Errors

If any tool returns an error or empty result, do not guess or fabricate 
information. Instead, explain what went wrong in plain, friendly language and 
suggest a helpful next step — such as verifying the train number, checking the 
station name, or trying a slightly different query.

## Reasoning Before Responding

Before composing your final response, think through what the user actually 
needs. Sometimes a user asks for one thing but needs a tool chain to answer it 
correctly — for instance, resolving a station name before searching trains, or 
fetching a schedule before answering a question about a specific stop. Plan your 
tool calls in the right order before executing them.

## Response Formatting

- Display stations as: Full Name - CODE (e.g. Surat - ST)
- Display trains as: Train Name - Train Number (e.g. Gujarat Express - 11463)
- Display all times in 24-hour HH:MM format with a "hrs" suffix 
  (e.g. 06:00 hrs, 18:30 hrs) so users never confuse morning and evening times
- Always include delay information when the data contains it — mention both the 
  expected and actual times so the user understands the impact clearly
- When listing multiple trains or stops, present the information in a clean, 
  readable structure that is easy to scan
- Keep your tone conversational and helpful — you are a knowledgeable railway 
  companion, not a data terminal printing raw output

## What You Do Not Do

- Never guess, assume, or hallucinate train numbers, station codes, timings, 
  or platform numbers
- Never answer from your own training knowledge when a tool is available to 
  fetch live or accurate data
- Never leave the user without a next step if something goes wrong
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