from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "env_parse_none_str": "null"}

    # OpenAI-compatible API (DeepSeek, OpenAI, etc.)
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-v4-pro"

    # Embedding (local via FastEmbed)
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_name: str = "opsmind_chunks"
    milvus_dim: int = 384  # must match embedding model dim

    # Data
    sample_data_dir: str = "./sampledata/all_documents"
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Database
    db_backend: str = "sqlite"       # "sqlite" or "postgres"
    sqlite_path: str = "./data/opsmind.db"
    postgres_dsn: str = "postgresql://localhost:5432/opsmind"

    # MCP Servers
    mcp_servers: list[dict] = []

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Demo: subset of categories to index (comma-separated env: DEMO_CATEGORIES)
    demo_categories_raw: str = "confluence,github"

    # Demo: max docs per category
    demo_max_docs_per_category: int = 50

    # Retrieval
    top_k: int = 5
    max_iterations: int = 3

    @property
    def demo_categories(self) -> list[str]:
        val = self.demo_categories_raw.strip()
        if not val:
            return []
        return [c.strip() for c in val.split(",") if c.strip()]

    @property
    def sample_data_path(self) -> Path:
        return Path(self.sample_data_dir)


settings = Settings()
