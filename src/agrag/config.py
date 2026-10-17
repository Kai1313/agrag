"""Configuration settings for AGRAG."""

from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # DeepSeek Configuration
    deepseek_api_key: Optional[str] = Field(default=None, alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL"
    )
    deepseek_default_model: Literal["deepseek-chat", "deepseek-reasoner"] = Field(
        default="deepseek-chat", alias="DEEPSEEK_DEFAULT_MODEL"
    )

    # Search Configuration
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")

    # Vector Store Configuration
    chroma_persist_dir: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIR"
    )
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    # Engine Defaults
    max_retries: int = Field(default=2, alias="MAX_RETRIES")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")

    @property
    def is_mock(self) -> bool:
        """Returns True if mock_mode is explicitly set or if no DEEPSEEK_API_KEY is provided."""
        return self.mock_mode or not bool(self.deepseek_api_key)


settings = Settings()
