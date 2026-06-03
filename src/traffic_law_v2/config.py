from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "staging", "prod"] = Field(default="dev", alias="APP_ENV")
    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8010, alias="APP_PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    model_provider: Literal["openai", "openai_compatible", "ollama", "azure_openai", "anthropic"] = Field(
        default="openai", alias="MODEL_PROVIDER"
    )
    llm_model: str = Field(default="google/gemini-3.1-flash-lite", alias="LLM_MODEL")
    embedding_provider: Literal["local", "openai", "openai_compatible", "hash"] = Field(
        default="local", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str | None = Field(default=None, alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=4, alias="EMBEDDING_BATCH_SIZE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434/v1", alias="OLLAMA_BASE_URL")
    ollama_api_key: str = Field(default="ollama", alias="OLLAMA_API_KEY")
    ollama_num_gpu: int = Field(default=0, alias="OLLAMA_NUM_GPU")
    ollama_num_ctx: int = Field(default=1024, alias="OLLAMA_NUM_CTX")
    ollama_num_batch: int = Field(default=32, alias="OLLAMA_NUM_BATCH")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1200, alias="LLM_MAX_TOKENS")

    database_url: str = Field(
        default="postgresql+psycopg://postgres:123456@localhost:5432/traffic_rag_v2",
        alias="DATABASE_URL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
