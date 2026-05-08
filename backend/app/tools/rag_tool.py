from langchain_core.tools import tool
from app.rag.retriever import retrieve_documents

@tool
def search_faq_policy_docs(query: str) -> str:
    """Search the FAQ/policy database and return a concise markdown string.
    The function caps total characters (~3000) to stay well below the LLM token limit.
    """
    # Retrieve a small number of top‑k documents
    docs = retrieve_documents(query, top_k=5)
    if not docs:
        return "No relevant documents found."

    MAX_TOTAL_CHARS = 3000  # roughly 750 tokens
    accumulated = []
    total = 0
    for i, doc in enumerate(docs, 1):
        snippet = doc.page_content.strip()[:600]  # per‑doc cap
        source = doc.metadata.get("source", "Unknown Source")
        entry = f"--- Document {i} ---\nSource: {source}\nContent:\n{snippet}\n"
        if total + len(entry) > MAX_TOTAL_CHARS:
            break
        accumulated.append(entry)
        total += len(entry)
    return "\n".join(accumulated)
        
    return "\n".join(formatted_docs)
