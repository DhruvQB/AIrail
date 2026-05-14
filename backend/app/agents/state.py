from typing import Annotated, Optional, Literal, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# ---------------------------------------------------------------------------
# Why TypedDict and not Pydantic BaseModel here?
#
# LangGraph's StateGraph requires TypedDict as the state type.
# The `Annotated[list[BaseMessage], add_messages]` pattern is LangGraph's
# built-in message reducer — it appends new messages rather than replacing
# the whole list on every node update, which is essential for multi-turn
# conversational memory.
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    The single state object that travels through the entire LangGraph graph.
    Every node receives this, makes its changes, and returns it.

    Fields are grouped by which layer owns/writes them:
      - Core fields:     set by the router (FastAPI) before graph.ainvoke()
      - Supervisor:      written by supervisor_node after intent classification
      - Agent layer:     written by whichever agent node runs
      - Output:          written by the agent as its final natural language answer
    """
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: Optional[Literal["rag", "railradar", "prs", "greeting", "guardrail", "offtopic"]]
    rag_output: Optional[dict[str, Any]]
    # Populated by rag_agent after Qdrant retrieval.
    # Shape: {
    #   "answer": str,           # synthesized answer from LLM
    #   "source_documents": [    # chunks retrieved from Qdrant
    #       {"content": str, "source": str, "score": float}
    #   ]
    # }

    railradar_output: Optional[dict[str, Any]]
    # Populated by railradar_agent after RailRadar API call.
    # Shape varies by tool called:
    #   get_live_status → {
    #       "train_number": str,
    #       "train_name": str,
    #       "current_station_code": str,
    #       "current_station_name": str,
    #       "status": str,
    #       "delay_minutes": int,
    #       "next_station_code": str | None,
    #       "next_station_name": str | None,
    #       "expected_arrival": str | None,
    #   }
    #   get_trains_between → {
    #       "source": str,
    #       "destination": str,
    #       "journey_date": str,
    #       "trains": [ { ...TrainSummary fields... } ]
    #   }

    prs_output: Optional[dict[str, Any]]
    # Populated by prs_agent after mock PNR or seat API call.
    # Shape varies by tool called:
    #   check_pnr → {
    #       "pnr_number": str,
    #       "train_number": str,
    #       "train_name": str,
    #       "journey_date": str,
    #       "source_station": str,
    #       "destination_station": str,
    #       "class_type": str,
    #       "chart_prepared": bool,
    #       "passengers": [ { ...PassengerStatus fields... } ]
    #   }
    #   check_seat_availability → {
    #       "train_number": str,
    #       "train_name": str,
    #       "source": str,
    #       "destination": str,
    #       "journey_date": str,
    #       "class_type": str,
    #       "quotas": [ { ...QuotaAvailability fields... } ]
    #   }

    response: Optional[str]
    error: Optional[str]
    error_code: Optional[str]