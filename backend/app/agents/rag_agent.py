"""
agents/rag_agent.py

LangGraph node that handles FAQ, policy, and rule queries for Indian Railways.
Retrieves all relevant chunks from Qdrant and passes them fully to the LLM.
"""

import os
import logging

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.rag.retriever import retrieve_documents
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM — no max_tokens cap; let the model respond as fully as needed
# ---------------------------------------------------------------------------

llm = ChatGroq(
    model=settings.LLM_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledgeable Indian Railways and IRCTC assistant.

You have access to official IRCTC documents including:
- Cancellation and refund rules
- E-ticketing service guidelines and terms of use
- Tatkal booking rules
- Frequently asked questions

When answering:
- Use the provided context documents as your primary source of truth.
- Give complete, detailed answers — do not truncate or summarize unnecessarily.
- Use Markdown formatting: bullet points, bold text, tables where helpful.
- If the context contains specific charges, percentages, or time windows, include them verbatim.
- Cite the source document name at the end of your response.
- If the context does not fully answer the question, say so and provide the best guidance you can from the documents available.
"""

# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

async def rag_agent_node(state: AgentState):
    """
    1. Extract the latest user query from state["messages"].
    2. Retrieve top-8 relevant chunks from Qdrant — no char limit cuts.
    3. Build a prompt with full context and call the LLM.
    """
    try:
        # --- Get latest user message ---
        latest_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                latest_query = msg.content
                break
            elif hasattr(msg, "role") and msg.role == "human":
                latest_query = msg.content
                break

        if not latest_query:
            latest_query = state["messages"][-1].content

        # --- Retrieve documents (top-8, full content, no arbitrary char cap) ---
        docs = retrieve_documents(latest_query, top_k=8)
        logger.info(f"[RAG] Retrieved {len(docs)} documents for query: {latest_query[:80]}")

        if not docs:
            context = "No relevant documents found in the knowledge base."
        else:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content.strip()
                raw_source = doc.metadata.get("source", "unknown")
                source = os.path.basename(raw_source) if raw_source else "unknown"
                context_parts.append(f"--- Document {i} | Source: {source} ---\n{content}")
            context = "\n\n".join(context_parts)

        # --- Build prompt ---
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context Documents:\n\n{context}\n\n---\n\nUser Question: {latest_query}"),
        ]

        # --- LLM call ---
        response = await llm.ainvoke(messages)
        response_text = response.content

        logger.info(f"[RAG] Response length: {len(response_text)} chars")

        return {
            **state,
            "messages": state["messages"] + [response],
            "response": response_text,
            "error": None,
            "error_code": None,
        }

    except Exception as e:
        error_msg = f"RAG agent encountered an error: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "response": None,
            "error": error_msg,
            "error_code": "AGENT_ERROR",
        }
