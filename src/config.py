# -*- coding: utf-8 -*-
"""
Configuration Module for Songkhla Old Town Hybrid GraphRAG System.
Handles environment variables, filesystem paths, model parameters, and LINE credentials.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static" / "images"

# Load .env
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class PathConfig:
    data_dir: Path = DATA_DIR
    chunks_path: Path = DATA_DIR / "songkhla_rag_chunks.json"
    facts_path: Path = DATA_DIR / "songkhla_places_facts.json"
    faiss_index_path: Path = DATA_DIR / "songkhla_faiss.index"
    bm25_index_path: Path = DATA_DIR / "songkhla_bm25.pkl"
    graph_pkl_path: Path = DATA_DIR / "songkhla_graph.pkl"
    triples_path: Path = DATA_DIR / "songkhla_knowledge_triples.json"
    schema_path: Path = DATA_DIR / "songkhla_ontology_schema.json"
    static_images_dir: Path = STATIC_DIR


@dataclass(frozen=True)
class ModelConfig:
    # Dense Embedding
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "kornwtp/ConGen-model-wangchanberta")
    
    # Cloud API LLM (Groq)
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    # Local LLM (Ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    primary_local_llm: str = os.getenv("DEFAULT_LOCAL_LLM", "qwen2.5:3b")
    comparison_local_llm: str = os.getenv("COMPARISON_LOCAL_LLM", "llama3.2:3b")
    
    # Graph Database (Neo4j)
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass: str = os.getenv("NEO4J_PASSWORD", "password1234")


@dataclass(frozen=True)
class RetrievalConfig:
    # RRF (Reciprocal Rank Fusion) parameters
    rrf_k: int = 60
    dense_weight: float = 0.45
    sparse_weight: float = 0.30
    graph_weight: float = 0.25
    
    # Dynamic Top-k bounds
    min_top_k: int = 2
    max_top_k: int = 7
    default_top_k: int = 4
    
    # Max prompt context token budget
    max_context_tokens: int = 2500


@dataclass(frozen=True)
class LineConfig:
    channel_secret: str = os.getenv("LINE_CHANNEL_SECRET", "")
    channel_access_token: str = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
    port: int = int(os.getenv("PORT", "5000"))
    base_image_url: str = os.getenv("BASE_IMAGE_URL", "http://localhost:5000/static/images")


paths = PathConfig()
models = ModelConfig()
retrieval = RetrievalConfig()
line_config = LineConfig()
