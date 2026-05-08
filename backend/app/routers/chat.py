"""
routers/chat.py
POST /api/chat

Entry point for all user messages. Routes through the LangGraph supervisor
which classifies intent and dispatches to the correct agent (RAG, RailRadar, PRS).
"""
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.models.schemas import ChatRequest, ChatResponse
from app.agents.graph import agent_graph

router = APIRouter()


@router.post(
    "/chat",
    summary="Send a message to the AI assistant",
    response_model=ChatResponse,
)
async def chat(params: ChatRequest):
    try:
        initial_state = {
            "messages": [HumanMessage(content=params.message)],
            "session_id": params.session_id,
            "intent": None,
            "rag_output": None,
            "railradar_output": None,
            "prs_output": None,
            "response": None,
            "error": None,
            "error_code": None,
        }

        result = await agent_graph.ainvoke(initial_state)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        return ChatResponse(
            success=True,
            message="OK",
            session_id=params.session_id,
            intent=result.get("intent", "unknown"),
            response=result.get("response", ""),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
