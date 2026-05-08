"""
routers/chat.py

POST /api/chat          — Send message, persist to DB, return AI response
GET  /api/chat/sessions — Get all sessions for logged-in user
GET  /api/chat/sessions/{session_id}/messages — Get messages for a session
DELETE /api/chat/sessions/{session_id} — Delete a session
"""
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas import ChatRequest, ChatResponse
from app.models.db_models import Session, Message
from app.agents.graph import agent_graph
from app.database import get_db
from app.routers.auth import get_current_user
from app.models.db_models import User

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /api/chat — send message and persist
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    summary="Send a message to the AI assistant",
    response_model=ChatResponse,
)
async def chat(
    params: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        # --- Ensure session exists in DB ---
        session_id = params.session_id
        result = await db.execute(
            select(Session).where(
                Session.id == session_id,
                Session.user_id == current_user.id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            # Create a new session row
            session = Session(id=session_id, user_id=current_user.id)
            db.add(session)
            await db.flush()

        # --- Persist user message ---
        user_msg_row = Message(
            session_id=session_id,
            role="user",
            content=params.message,
        )
        db.add(user_msg_row)
        await db.flush()

        # --- Run AI graph ---
        initial_state = {
            "messages": [HumanMessage(content=params.message)],
            "session_id": session_id,
            "intent": None,
            "rag_output": None,
            "railradar_output": None,
            "prs_output": None,
            "response": None,
            "error": None,
            "error_code": None,
        }

        graph_result = await agent_graph.ainvoke(initial_state)

        if graph_result.get("error"):
            raise HTTPException(status_code=500, detail=graph_result["error"])

        response_text = graph_result.get("response", "")

        # --- Persist assistant response ---
        assistant_msg_row = Message(
            session_id=session_id,
            role="assistant",
            content=response_text,
        )
        db.add(assistant_msg_row)
        # commit happens automatically in get_db

        return ChatResponse(
            success=True,
            message="OK",
            session_id=session_id,
            intent=graph_result.get("intent", "unknown"),
            response=response_text,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Chat] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# GET /api/chat/sessions — list all sessions for the logged-in user
# ---------------------------------------------------------------------------

@router.get("/chat/sessions", summary="Get all chat sessions for the logged-in user")
async def get_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Session)
        .where(Session.user_id == current_user.id)
        .order_by(Session.created_at.desc())
    )
    sessions = result.scalars().all()

    session_list = []
    for s in sessions:
        # Fetch first message to use as title/preview
        msg_result = await db.execute(
            select(Message)
            .where(Message.session_id == s.id)
            .order_by(Message.timestamp.asc())
            .limit(1)
        )
        first_msg = msg_result.scalar_one_or_none()
        title = first_msg.content[:50] if first_msg else "New Chat"
        preview = first_msg.content[:80] if first_msg else ""

        session_list.append({
            "sessionId": str(s.id),
            "title": title,
            "preview": preview,
            "createdAt": s.created_at.isoformat(),
        })

    return {"success": True, "sessions": session_list}


# ---------------------------------------------------------------------------
# GET /api/chat/sessions/{session_id}/messages — get messages in a session
# ---------------------------------------------------------------------------

@router.get(
    "/chat/sessions/{session_id}/messages",
    summary="Get full message history for a session",
)
async def get_session_messages(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Verify session belongs to this user
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.timestamp.asc())
    )
    messages = msg_result.scalars().all()

    return {
        "success": True,
        "sessionId": session_id,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
            }
            for m in messages
        ],
    }


# ---------------------------------------------------------------------------
# DELETE /api/chat/sessions/{session_id} — delete a session and its messages
# ---------------------------------------------------------------------------

@router.delete(
    "/chat/sessions/{session_id}",
    summary="Delete a chat session and all its messages",
)
async def delete_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Session).where(
            Session.id == session_id,
            Session.user_id == current_user.id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    await db.delete(session)
    return {"success": True, "message": "Session deleted"}
