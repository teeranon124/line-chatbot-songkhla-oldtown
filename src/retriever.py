import os
import json
import pickle
from typing import List, Dict, Any

# Ensure instantaneous offline loading from local cache
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import faiss
import numpy as np
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

from .config import paths, models, retrieval
from .graph_engine import SongkhlaGraphEngine


class DenseRetriever:
    def __init__(self, model_name: str = models.embedding_model_name):
        self.model_name = model_name
        print(f"[DenseRetriever] Initializing {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.load_index(paths.faiss_index_path)

    def load_index(self, index_path):
        if os.path.exists(index_path):
            self.index = faiss.read_index(str(index_path))
            print(f"[DenseRetriever] Loaded FAISS index with {self.index.ntotal} vectors.")
        else:
            print(f"[DenseRetriever Warning] Index not found at {index_path}.")

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.index is None:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(q_emb, k=top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({"chunk_index": int(idx), "score": float(score)})
        return results


class SparseRetriever:
    def __init__(self):
        self.bm25 = None
        self.load_index(paths.bm25_index_path)

    def load_index(self, bm25_path):
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
            print("[SparseRetriever] Loaded BM25 index successfully.")
        else:
            print(f"[SparseRetriever Warning] BM25 file not found at {bm25_path}.")

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokens = word_tokenize(query, engine="newmm")
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append({"chunk_index": int(idx), "score": score})
        return results


class DynamicTopKManager:
    """Calculates optimal Top-K (2, 4, 7) based on empirical query complexity."""
    @staticmethod
    def calculate_k(query: str, default_k: int = retrieval.default_top_k) -> int:
        q_clean = query.strip()
        q_len = len(q_clean)
        
        # 1. Complex Multi-constraint / Itinerary / Budget planning (Need k=7 to prevent context starvation)
        complex_kws = ["ทริป", "ตาราง", "วางแผน", "ทั้งวัน", "2 วัน", "1 วัน", "งบ", "งบประมาณ", "คำนวณ", "กี่บาท", "เส้นทางเดิน"]
        if any(kw in q_clean for kw in complex_kws) or (q_len > 55 and "และ" in q_clean):
            return retrieval.max_top_k  # 7

        # 2. Simple Fact-seeking (Hours, phone, location) -> k=2 (Minimize tokens and reduce distraction)
        simple_fact_kws = ["กี่โมง", "เปิดกี่โมง", "ปิดกี่โมง", "เบอร์โทร", "เบอร์", "โทรศัพท์", "ที่ตั้ง", "พิกัด", "ตั้งอยู่ถนน"]
        if any(kw in q_clean for kw in simple_fact_kws) or q_len < 22:
            return retrieval.min_top_k  # 2

        # 3. Standard Comparison / Area discovery queries -> k=4
        return default_k  # 4


class HybridRetriever:
    """Orchestrates Dense, Sparse, and Graph retrievers with RRF Fusion."""

    def __init__(self, graph_engine: SongkhlaGraphEngine = None):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.graph = graph_engine or SongkhlaGraphEngine()
        
        # Load metadata chunks
        with open(paths.chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

    def retrieve(self, query: str, top_k: int = None, mode: str = "hybrid") -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = DynamicTopKManager.calculate_k(query)

        # 1. Dense Only
        if mode == "dense":
            dense_res = self.dense.search(query, top_k=top_k)
            return [self.chunks[r["chunk_index"]] for r in dense_res if r["chunk_index"] < len(self.chunks)]

        # 2. Sparse Only
        if mode == "sparse":
            sparse_res = self.sparse.search(query, top_k=top_k)
            return [self.chunks[r["chunk_index"]] for r in sparse_res if r["chunk_index"] < len(self.chunks)]

        # 3. Graph Only
        if mode == "graph":
            return self.graph.search_subgraph(query, top_k=top_k)

        # 4. Tri-Hybrid with Reciprocal Rank Fusion (RRF)
        rrf_k = retrieval.rrf_k
        w_dense = retrieval.dense_weight
        w_sparse = retrieval.sparse_weight
        w_graph = retrieval.graph_weight

        fused_scores = {}
        chunk_map = {}

        # Dense candidates
        dense_hits = self.dense.search(query, top_k=top_k * 2)
        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_index"]
            if cid < len(self.chunks):
                chunk = self.chunks[cid]
                key = chunk["chunk_id"]
                chunk_map[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + w_dense / (rrf_k + rank + 1)

        # Sparse candidates
        sparse_hits = self.sparse.search(query, top_k=top_k * 2)
        for rank, hit in enumerate(sparse_hits):
            cid = hit["chunk_index"]
            if cid < len(self.chunks):
                chunk = self.chunks[cid]
                key = chunk["chunk_id"]
                chunk_map[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + w_sparse / (rrf_k + rank + 1)

        # Build mapping from place_id to chunk
        place_to_chunk = {}
        for c in self.chunks:
            pid = c.get("place_id")
            if pid:
                place_to_chunk[pid] = c

        # Graph candidates
        graph_hits = self.graph.search_subgraph(query, top_k=top_k * 2)
        for rank, g_chunk in enumerate(graph_hits):
            raw_id = g_chunk.get("chunk_id", "").replace("graph_", "")
            matched_chunk = place_to_chunk.get(raw_id)
            if matched_chunk:
                key = matched_chunk["chunk_id"]
                if key not in chunk_map:
                    chunk_map[key] = dict(matched_chunk)
                # Augment text chunk with Knowledge Graph relational evidence
                chunk_map[key]["graph_subgraph"] = g_chunk.get("content", "")
            else:
                key = g_chunk["chunk_id"]
                chunk_map[key] = g_chunk
            fused_scores[key] = fused_scores.get(key, 0.0) + w_graph / (rrf_k + rank + 1)

        # Sort by fused score
        sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
        final_chunks = [chunk_map[k] for k in sorted_keys[:top_k]]
        return final_chunks
