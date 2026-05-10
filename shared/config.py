from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str
    groq_api_key: str = ""  # only required for Part 2 chatbot

    data_path: Path = Path("sample_tickets_v6.json_")
    cache_dir: Path = Path(".cache")
    chroma_dir: Path = Path(".chroma")

    groq_model: str = "llama-3.3-70b-versatile"

    # Clustering
    umap_n_components: int = 15
    umap_n_neighbors: int = 15
    hdbscan_min_cluster_size: int = 250   # 11 clusters, 41% noise (from sweep)
    hdbscan_min_samples: int = 10
    max_clusters_to_label: int = 15        # cap LLM calls for labeling
    min_new_cluster_tickets: int = 20

    # RAG
    top_k_retrieval: int = 5


settings = Settings()
