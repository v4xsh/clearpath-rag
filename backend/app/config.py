from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    groq_api_key: str
    faiss_index_path: str = "backend/data/faiss.index"
    bm25_corpus_path: str = "backend/data/bm25_corpus.pkl"
    metadata_path: str = "backend/data/metadata.json"
    embedding_model: str = "all-MiniLM-L6-v2"
    log_level: str = "INFO"

    # Retrieval
    top_k_simple: int = 3
    top_k_medium: int = 5
    top_k_complex: int = 8

    # Sufficiency thresholds
    coverage_high_threshold: float = 0.65
    coverage_low_threshold: float = 0.35

    # Fusion weights
    w_sem_high_density: float = 0.60
    w_bm25_high_density: float = 0.40
    w_sem_default: float = 0.75
    w_bm25_default: float = 0.25

    # Memory
    memory_token_budget: int = 2048

    # Models
    model_simple: str = "llama-3.1-8b-instant"
    model_complex: str = "llama-3.3-70b-versatile"

    # Router thresholds
    simple_max_words: int = 12

    model_config = {
        "protected_namespaces": (),
        "env_file": "backend/.env",
        "env_file_encoding": "utf-8",
    }

    @field_validator(
        "faiss_index_path",
        "bm25_corpus_path",
        "metadata_path",
        mode="before",
    )
    @classmethod
    def make_absolute(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            p = BASE_DIR / v
        return str(p)


settings = Settings()