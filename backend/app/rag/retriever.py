from qdrant_client import QdrantClient, models
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document

from app.rag.embedding import get_embedding, get_sparse_embedding

# Initialize FlashRank ranker (loads model on import or first use)
ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

def retrieve_documents(query: str, top_k: int = 5) -> list[Document]:
    """
    Performs a hybrid search (Dense + Sparse) on Qdrant using RRF,
    fetches the top results, and uses FlashRank to rerank them.
    """
    client = QdrantClient(
        host="localhost",
        port=6334,
        prefer_grpc=True
    )

    dense_vector = get_embedding().embed_query(query)
    sparse_vector = get_sparse_embedding().embed_query(query)

    # 1. FAST HYBRID SEARCH (NO PAYLOAD)
    search_result = client.query_points(
        collection_name="AIrail",
        prefetch=[
            models.Prefetch(query=dense_vector, using="dense", limit=30),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices,
                    values=sparse_vector.values
                ),
                using="sparse",
                limit=30
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=20,
        with_payload=False
    )

    hits = search_result.points

    if not hits:
        return []

    # 2. FETCH ONLY TOP N FOR RERANK
    top_ids = [hit.id for hit in hits]

    payload_docs = client.retrieve(
        collection_name="AIrail",
        ids=top_ids,
        with_payload=True
    )

    # 3. PREPARE FOR RERANK (LIMIT SIZE)
    passages = [
        {
            "id": str(d.id),
            "text": d.payload.get("page_content", ""),
            "meta": d.payload
        }
        for d in payload_docs
    ]

    # FlashRank request
    rerank_request = RerankRequest(query=query, passages=passages[:20])
    reranked_results = ranker.rerank(rerank_request)

    final_passages = reranked_results[:top_k]

    # 4. CONVERT TO DOCUMENTS
    docs = [
        Document(
            page_content=p["text"],
            metadata=p["meta"].get("metadata", {})
        )
        for p in final_passages
    ]

    return docs
