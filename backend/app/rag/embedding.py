# Embedding model (e.g. sentence-transformers)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse

def get_embedding():
    return FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs = {
            "normalize_embeddings": True,
            "batch_size": 128,
            "parallelize": True,
            "parallel" : 0
        }
    )

def get_sparse_embedding():
    return FastEmbedSparse(
        model_name="Qdrant/bm25"
    )