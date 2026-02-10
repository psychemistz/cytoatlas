"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "CytoAtlas API"
    app_version: str = "0.1.0"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    @field_validator("environment", mode="before")
    @classmethod
    def normalize_environment(cls, v: str) -> str:
        """Normalize environment value, mapping HPC values to standard ones."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'").lower()
            # Map HPC-specific values
            if v in ("batch", "prod", "prd"):
                return "production"
            if v in ("dev", "local"):
                return "development"
            if v in ("stage", "stg"):
                return "staging"
        return v

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, v) -> bool:
        """Normalize debug value, handling quoted strings."""
        if isinstance(v, str):
            v = v.strip().strip('"').strip("'").lower()
            return v in ("true", "1", "yes", "on")
        return bool(v)

    @field_validator("api_v1_prefix", mode="before")
    @classmethod
    def normalize_api_prefix(cls, v: str) -> str:
        """Strip quotes from API prefix."""
        if isinstance(v, str):
            return v.strip().strip('"').strip("'")

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """Validate security settings in production."""
        if self.environment == "production":
            if self.secret_key is None or self.secret_key == "change-me-in-production-use-openssl-rand-hex-32":
                raise ValueError(
                    "SECRET_KEY must be set to a secure value in production. "
                    "Generate one with: openssl rand -hex 32"
                )
        return self

    # API
    api_v1_prefix: str = "/api/v1"
    api_v2_prefix: str = "/api/v2"
    allowed_origins: str = Field(default="http://localhost:8000,http://localhost:3000")
    max_request_body_mb: int = 100

    @property
    def cors_origins(self) -> list[str]:
        """Get CORS origins as list."""
        if self.allowed_origins == "*":
            return ["*"]
        return [o.strip() for o in self.allowed_origins.split(",")]

    # Database (optional - leave empty for no database)
    database_url: str | None = Field(default=None)
    database_pool_size: int = 5
    database_max_overflow: int = 10
    database_pool_timeout: int = 30

    # Redis (optional - leave empty for in-memory cache)
    redis_url: str | None = Field(default=None)
    redis_cache_ttl: int = 3600  # 1 hour default

    @property
    def use_database(self) -> bool:
        """Check if database is configured."""
        return bool(self.database_url)

    @property
    def use_redis(self) -> bool:
        """Check if Redis is configured."""
        return bool(self.redis_url)

    # Authentication
    secret_key: str | None = Field(default=None)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    api_key_header: str = "X-API-Key"
    require_auth: bool = False
    audit_enabled: bool = True
    audit_log_path: Path = Field(default=Path("logs/audit.jsonl"))

    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    trusted_proxies: str = ""  # Comma-separated CIDR ranges (e.g., "10.0.0.0/8,172.16.0.0/12")

    # Data Paths
    h5ad_base_path: Path = Field(default=Path("/data/Jiang_Lab/Data/Seongyong"))
    results_base_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/results")
    )
    viz_data_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/visualization/data")
    )

    # DuckDB (primary data backend — replaces JSON/Parquet/SQLite scatter)
    duckdb_atlas_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/atlas_data.duckdb"),
        description="Path to DuckDB file containing all science data",
    )

    # SQLite for app state (users, conversations, jobs)
    sqlite_app_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/app.db"),
        description="Path to SQLite file for app state (users, chat, jobs)",
    )

    # Legacy data paths (deprecated — kept for fallback when DuckDB unavailable)
    parquet_data_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/visualization/data/parquet")
    )
    sqlite_scatter_db_path: Path = Field(
        default=Path("/vf/users/parks34/projects/2secactpy/visualization/data/validation_scatter.db")
    )

    @property
    def use_duckdb(self) -> bool:
        """Check if DuckDB atlas file exists."""
        return self.duckdb_atlas_path.exists()

    # CIMA paths
    cima_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/"
            "CIMA_RNA_6484974cells_36326genes_compressed.h5ad"
        )
    )
    cima_biochem: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/"
            "CIMA_Sample_Blood_Biochemistry_Results.csv"
        )
    )
    cima_metabolites: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/"
            "CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv"
        )
    )

    # Inflammation Atlas paths
    inflammation_main_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
            "INFLAMMATION_ATLAS_main_afterQC.h5ad"
        )
    )
    inflammation_validation_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
            "INFLAMMATION_ATLAS_validation_afterQC.h5ad"
        )
    )
    inflammation_external_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/"
            "INFLAMMATION_ATLAS_external_afterQC.h5ad"
        )
    )

    # scAtlas paths
    scatlas_normal_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad"
        )
    )
    scatlas_cancer_h5ad: Path = Field(
        default=Path(
            "/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad"
        )
    )

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # File Upload
    max_upload_size_gb: int = Field(default=50)
    upload_dir: Path = Field(default=Path("/data/cytoatlas/uploads"))

    @property
    def max_upload_size_bytes(self) -> int:
        """Get max upload size in bytes."""
        return self.max_upload_size_gb * 1024 * 1024 * 1024

    # LLM API for Chat — primary: vLLM (OpenAI-compatible); fallback: Anthropic Claude
    llm_base_url: str | None = Field(default="http://localhost:8001/v1")
    llm_api_key: str = Field(default="not-needed")  # vLLM doesn't require auth by default
    chat_model: str = Field(default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    chat_max_tokens: int = Field(default=8192)
    # Anthropic fallback (used when llm_base_url is not set or vLLM is unreachable)
    anthropic_api_key: str | None = Field(default=None)
    anthropic_chat_model: str = Field(default="claude-sonnet-4-5-20250929")

    # Chat Rate Limiting
    anon_chat_limit_per_day: int = Field(default=5)
    auth_chat_limit_per_day: int = Field(default=1000)

    # RAG (Retrieval Augmented Generation)
    rag_enabled: bool = Field(default=True)
    rag_db_path: Path = Field(default=Path("rag_db"))
    rag_embedding_model: str = Field(default="all-MiniLM-L6-v2")
    rag_top_k: int = Field(default=5)

    @property
    def cima_results_dir(self) -> Path:
        return self.results_base_path / "cima"

    @property
    def inflammation_results_dir(self) -> Path:
        return self.results_base_path / "inflammation"

    @property
    def scatlas_results_dir(self) -> Path:
        return self.results_base_path / "scatlas"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
