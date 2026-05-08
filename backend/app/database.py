"""
database.py
Async SQLAlchemy engine + session factory.

Tables are created automatically on startup via init_db().
No migration tool needed for local development.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings


engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.APP_ENV == "development",
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """All ORM models inherit from this base."""
    pass


async def get_db() -> AsyncSession:
    """FastAPI dependency — yields a managed async DB session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Create all tables on startup if they don't already exist.
    Simple and idempotent — safe to call every time the app starts.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
