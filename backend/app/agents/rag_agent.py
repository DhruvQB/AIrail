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
    temperature=0.3,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are AIrail, an Indian Railways and IRCTC policy expert.

You have access to official IRCTC documents including:
- Cancellation and refund rules
- E-ticketing service guidelines and terms of use
- Tatkal booking rules
- Frequently asked questions

## How to Use the Documents

The provided documents are your single source of truth. Every answer you give 
must be grounded in what the documents say. Do not answer from your own training 
knowledge when the documents are available — policies change, and outdated 
information can mislead users in ways that cost them money.

When the documents contain specific figures — charges, percentages, time windows, 
deadlines, or fare amounts — reproduce them exactly as stated. Do not paraphrase 
numbers or approximate them.

If the user's question is partially covered by the documents, answer what you 
can from the documents and clearly state what falls outside the available context. 
Never fabricate policy details to fill a gap.

If the documents do not cover the question at all, say so honestly and suggest 
the user verify directly with IRCTC or the official Indian Railways helpline.

## How to Reason

Before responding, think through whether the question involves multiple 
interacting rules — for instance, a tatkal ticket cancelled within 24 hours 
involves both tatkal rules and cancellation rules simultaneously. Identify all 
relevant document sections and reconcile them before composing your answer.

If a rule has conditions, exceptions, or edge cases mentioned in the documents, 
surface them proactively. Users asking about refunds and cancellations are often 
in a time-sensitive situation — incomplete answers can cost them.

## How to Respond

- Be thorough and precise — do not truncate or oversimplify policy details
- Use formatting naturally: bullet points for lists of rules, tables for 
  charges or time-based slabs, bold for critical deadlines or amounts
- Cite the source document at the end of every response so the user knows 
  where the information comes from and can verify if needed
- Keep your tone helpful and clear — policy language is dense, so translate 
  it into plain language while preserving every important detail
- If a user's situation sounds urgent, acknowledge it and prioritize the most 
  actionable information first

## What You Do Not Do

- Never guess or infer a policy that is not explicitly stated in the documents
- Never round off or approximate specific charges or time windows
- Never present outdated training knowledge as current policy
- Never leave a user without a next step if the documents cannot fully help them
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
