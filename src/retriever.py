import os
import json
import pickle
from typing import List, Dict, Any

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

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
    """
    Mathematical Score-Based Adaptive Top-K Selector (Knee-point / Score Cliff Cutoff).
    Does NOT use arbitrary keyword matching.
    Analyzes the relative drop-off (gradient) and score distribution of candidate hits.
    """
    @staticmethod
    def select_adaptive_k(
        ranked_scores: List[float],
        min_k: int = 2,
        max_k: int = 7,
        drop_ratio_threshold: float = 0.65,
        knee_drop_delta: float = 0.25
    ) -> int:
        """
        Calculates optimal cutoff rank k purely from retrieval score distribution:
        - If candidate 1 or 2 is dominant and there is a steep cliff drop (knee point),
          cuts off early (k=2) to avoid injecting irrelevant noise.
        - If scores remain clustered (broad / multi-hop query),
          retains candidates up to max_k (e.g. k=4 to 7).
        """
        if not ranked_scores or len(ranked_scores) <= min_k:
            return min(len(ranked_scores) if ranked_scores else min_k, min_k)

        top_score = ranked_scores[0]
        if top_score <= 1e-6:
            return min_k

        cutoff_k = min_k
        for i in range(1, min(len(ranked_scores), max_k)):
            curr_score = ranked_scores[i]
            prev_score = ranked_scores[i - 1]
            
            ratio_to_top = curr_score / top_score
            step_drop = (prev_score - curr_score) / top_score

            if step_drop >= knee_drop_delta or ratio_to_top < drop_ratio_threshold:
                cutoff_k = max(i, min_k)
                break
            cutoff_k = i + 1

        return min(max(cutoff_k, min_k), max_k)

    @staticmethod
    def calculate_k(query: str, default_k: int = retrieval.default_top_k) -> int:
        """Fallback for backward compatibility."""
        return default_k


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
        pool_k = retrieval.max_top_k * 2

        # 1. Dense Only
        if mode == "dense":
            dense_res = self.dense.search(query, top_k=pool_k)
            scores = [r["score"] for r in dense_res]
            k = top_k if top_k is not None else DynamicTopKManager.select_adaptive_k(scores, min_k=retrieval.min_top_k, max_k=retrieval.max_top_k)
            return [self.chunks[r["chunk_index"]] for r in dense_res[:k] if r["chunk_index"] < len(self.chunks)]

        # 2. Sparse Only
        if mode == "sparse":
            sparse_res = self.sparse.search(query, top_k=pool_k)
            scores = [r["score"] for r in sparse_res]
            k = top_k if top_k is not None else DynamicTopKManager.select_adaptive_k(scores, min_k=retrieval.min_top_k, max_k=retrieval.max_top_k)
            return [self.chunks[r["chunk_index"]] for r in sparse_res[:k] if r["chunk_index"] < len(self.chunks)]

        # 3. Graph Only
        if mode == "graph":
            k = top_k if top_k is not None else retrieval.default_top_k
            return self.graph.search_subgraph(query, top_k=k)

        # 4. Tri-Hybrid with Reciprocal Rank Fusion (RRF)
        rrf_k = retrieval.rrf_k
        w_dense = retrieval.dense_weight
        w_sparse = retrieval.sparse_weight
        w_graph = retrieval.graph_weight

        fused_scores = {}
        chunk_map = {}

        # Dense candidates
        dense_hits = self.dense.search(query, top_k=pool_k)
        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_index"]
            if cid < len(self.chunks):
                chunk = self.chunks[cid]
                key = chunk["chunk_id"]
                chunk_map[key] = chunk
                fused_scores[key] = fused_scores.get(key, 0.0) + w_dense / (rrf_k + rank + 1)

        # Sparse candidates
        sparse_hits = self.sparse.search(query, top_k=pool_k)
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
        graph_hits = self.graph.search_subgraph(query, top_k=pool_k)
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
        ranked_scores = [fused_scores[k] for k in sorted_keys]

        # Dynamically determine optimal cutoff k purely from score drop distribution
        effective_k = top_k if top_k is not None else DynamicTopKManager.select_adaptive_k(
            ranked_scores,
            min_k=retrieval.min_top_k,
            max_k=retrieval.max_top_k
        )

        final_chunks = [chunk_map[k] for k in sorted_keys[:effective_k]]
        return final_chunks
