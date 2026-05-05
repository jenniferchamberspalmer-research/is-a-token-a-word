from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="BOOK_", extra="ignore")

    data_dir: Path = Path("data")
    db_path: Path = Path("data/books.db")

    anthropic_api_key: str = ""
    chat_model: str = "claude-opus-4-7"
    overview_model: str = "claude-opus-4-7"

    chunk_target_chars: int = 1800
    chunk_overlap_chars: int = 200
    retrieval_k: int = 6

    stt_provider: str = "stub"
    tts_provider: str = "stub"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
