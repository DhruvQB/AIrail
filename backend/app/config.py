"""
config.py
Settings loaded via pydantic-settings.
All env vars are validated at startup.
"""
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    APP_ENV: str = Field(default="development")

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://airail_user:airail_pass@localhost:5432/airail_db"
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def fix_postgres_protocol(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    # ── Qdrant ─────────────────────────────────────────────────────────────
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_COLLECTION: str = Field(default="railway_docs")

    # ── External APIs ──────────────────────────────────────────────────────
    RAILRADAR_API_KEY: str = Field(default="your_railradar_key_here")
    RAILRADAR_BASE_URL: str = Field(default="https://api.railradar.org/api/v1")

    GROQ_API_KEY: str = Field(default="your_groq_key_here")
    LLM_MODEL: str = Field(default="llama3-70b-8192")

    # ── Auth / JWT ─────────────────────────────────────────────────────────
    JWT_SECRET_KEY: str = Field(default="change-me-in-production-airail-secret-key-2026")
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRE_MINUTES: int = Field(default=1440)  # 24 hours

    # ── Embeddings ─────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
