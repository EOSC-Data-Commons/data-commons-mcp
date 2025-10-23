"""Define the service settings and configurable parameters for the agent."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Define the service settings for the server that can be set using environment variables."""

    # Seach results settings
    opensearch_results_count: int = 5
    reranking_results_count: int = 5

    # Server settings
    server_port: int = 8000
    server_host: str = "0.0.0.0"  # noqa: S104
    cors_enabled: bool = True

    # OpenSearch settings
    opensearch_url: str = "http://localhost:9200"
    # opensearch_url: str = "http://opensearch:9200"
    opensearch_index: str = "test_datacite"

    # Embedding models: https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    # embedding_model: str = "BAAI/bge-base-en-v1.5"
    # embedding_dimensions: int = 768

    # LLM providers API keys
    default_llm_model: str = "einfracz/qwen3-coder"
    einfracz_api_key: str = ""
    openrouter_api_key: str = ""
    # default_max_tokens: int = 16384

    # The name of the application used for display
    app_name: str = "EOSC Data Commons MCP"
    # Public API key used by the frontend to access the chatbot and prevent abuse from bots
    chat_api_key: str = ""

    logs_filepath: str = "./data/logs/conversations.jsonl"

    model_config = SettingsConfigDict(
        env_file="keys.env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @property
    def server_url(self) -> str:
        """Computed server URL using the host and port, for accessing locally for /mcp calls.

        Returns:
            A string like 'http://0.0.0.0:8888'.
        """
        return f"http://{self.server_host}:{self.server_port}"


settings = Settings()
