"""
main.py
FastAPI application factory.

Lifecycle
---------
startup  → verify DB is reachable, log config summary
shutdown → dispose async engine cleanly
"""
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import engine, init_db
from app.routers import auth, availability, chat, pnr_status, search_trains, train_status
from app.rag.ingest import run_startup_ingestion


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ────────────────────────────────────────────────────────────
    print(f"[startup] ENV={settings.APP_ENV}  DB={settings.DATABASE_URL.split('@')[-1]}")
    await init_db()
    await asyncio.get_event_loop().run_in_executor(None, run_startup_ingestion)
    yield
    # ── shutdown ───────────────────────────────────────────────────────────
    await engine.dispose()
    print("[shutdown] DB engine disposed.")


app = FastAPI(
    title="AIrail — Railway AI Bot API",
    description=(
        "FastAPI backend powering a LangGraph multi-agent system for Indian Railways queries. "
        "Handles live train status, seat availability, PNR tracking, and FAQ via RAG."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ───────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel and all other origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────
app.include_router(auth.router,          prefix="/api", tags=["Auth"])
app.include_router(chat.router,          prefix="/api", tags=["Chat"])
app.include_router(train_status.router,  prefix="/api", tags=["Train Status"])
app.include_router(search_trains.router, prefix="/api", tags=["Search Trains"])
app.include_router(availability.router,  prefix="/api", tags=["Availability"])
app.include_router(pnr_status.router,    prefix="/api", tags=["PNR Status"])


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"], summary="Root health check")
async def root():
    return {"status": "ok", "version": app.version, "env": settings.APP_ENV}


@app.get("/health", tags=["Health"], summary="Detailed health check")
async def health():
    return {
        "status": "ok",
        "database": settings.DATABASE_URL.split("@")[-1],  # hide credentials
        "qdrant": settings.QDRANT_URL,
    }
