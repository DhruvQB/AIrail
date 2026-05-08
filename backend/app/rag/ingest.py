from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance, SparseVectorParams

from app.rag.embedding import get_embedding, get_sparse_embedding

from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.embeddings import FastEmbedEmbeddings

import time
import os

DATA_DIR = r"C:\\Users\\Dhruv\\Desktop\\AIrail\\backend\\data"
INGESTED_TRACKER = os.path.join(DATA_DIR, ".ingested_files")

# Global Progress
progress_data = {
    "status": "idle",
    "progress": 0,
    "current_step": ""
}

# Qdrant Client
client = QdrantClient(
    host="localhost",
    port=6334,
    prefer_grpc=True
)

COLLECTION_NAME = "AIrail"

dense_embedding = get_embedding()
sparse_embedding = get_sparse_embedding()
vector_store = None

# Ensure Collection Exists
def init_collection():
    global vector_store
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                    on_disk=True
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    #on_disk=True
                )
            },
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True
                )
            )
        )
        print("✅ Created collection AIrail")


# Load PDF
def load_document(file_path):
    loader = PyMuPDFLoader(file_path)
    return loader.load()


# Chunking Strategies
def split_document(docs):
    splitter = SemanticChunker(dense_embedding)
    return splitter.split_documents(docs)


# Worker Thread
def process_batch(batch, batch_id):

    start = time.time()

    db = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse"
    )

    db.add_documents(batch)

    end = time.time()

    print(f"⚡ Batch {batch_id}: {end-start:.2f} sec")


# Store Chunks
def create_vector_store(chunks, start_prog, end_prog):

    global progress_data

    batch_size = 256

    batches = [
        chunks[i:i+batch_size]
        for i in range(0, len(chunks), batch_size)
    ]

    if not batches:
        progress_data["progress"] = int(end_prog)
        return

    max_workers = 3

    print(f"🚀 Starting ingestion with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(process_batch, batch, idx + 1)
            for idx, batch in enumerate(batches)
        ]

        completed = 0

        for future in as_completed(futures):
            future.result()

            completed += 1

            progress_data["progress"] = min(
                int(
                    start_prog +
                    (end_prog - start_prog) *
                    (completed / len(batches))
                ),
                100
            )


# Main Ingest Pipeline
def ingest(file_paths):
    global progress_data
    progress_data["status"] = "ingesting"
    progress_data["progress"] = 0

    init_collection()
    total_files = len(file_paths)

    for i, file in enumerate(file_paths):
        file_base = 100.0 * (i / total_files)
        file_range = 100.0 / total_files
        filename = os.path.basename(file)

        # ---------------- LOAD ----------------
        progress_data["current_step"] = f"Loading {filename}"
        docs = load_document(file)

        # ---------------- CHUNK ----------------
        progress_data["current_step"] = f"Chunking {filename}"
        chunks = split_document(docs)

        # ---------------- STORE ----------------
        progress_data["current_step"] = f"Storing {filename}"
        create_vector_store(
            chunks,
            file_base + file_range * 0.2,
            file_base + file_range
        )
        
        # Mark as ingested
        with open(INGESTED_TRACKER, "a") as f:
            f.write(f"{filename}\n")
        
    progress_data["status"] = "completed"
    progress_data["progress"] = 100
    progress_data["current_step"] = "Done"

def run_startup_ingestion():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    all_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    
    if os.path.exists(INGESTED_TRACKER):
        with open(INGESTED_TRACKER, "r") as f:
            ingested = set(line.strip() for line in f)
    else:
        ingested = set()
        
    new_pdfs = [f for f in all_files if f not in ingested]
    
    if not new_pdfs:
        print("✅ No new PDFs to ingest. Startup ingestion skipped.")
        return
        
    print(f"🚀 Found {len(new_pdfs)} new PDFs for ingestion.")
    file_paths = [os.path.join(DATA_DIR, f) for f in new_pdfs]
    ingest(file_paths)