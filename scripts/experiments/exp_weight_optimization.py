# -*- coding: utf-8 -*-
"""
Experiment: Systematic Weight Sensitivity Optimization for Tri-Hybrid RRF.
Investigates the impact of Dense, Sparse, and Graph weights on Hit@1, Hit@3, and MRR.
Eliminates speculation by running empirical tests across 15 benchmark queries.
"""

import sys
import os
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.retriever import HybridRetriever
from scripts.experiments.exp2_retrieval_ablation import BENCHMARK_15_QUERIES, is_match

def run_weight_grid():
    retriever = HybridRetriever()
    pool_k = 14
    rrf_k = 60

    test_configs = [
        ("1. Baseline (Current)", 0.45, 0.30, 0.25),
        ("2. Equal Weights (1:1:1)", 0.333, 0.333, 0.334),
        ("3. Sparse-Boosted", 0.30, 0.45, 0.25),
        ("4. Balanced Sparse/Dense", 0.40, 0.40, 0.20),
        ("5. Sparse-Dominant", 0.25, 0.50, 0.25),
        ("6. Graph-Optimized", 0.35, 0.35, 0.30),
        ("7. High Sparse High Dense", 0.45, 0.45, 0.10)
    ]

    print("=" * 80)
    print("🔬 SYSTEMATIC WEIGHT SENSITIVITY EXPERIMENT (N=15 Queries)")
    print("=" * 80)
    header = f"{'Configuration Name':<28} | {'Dense':<6} {'Sparse':<6} {'Graph':<6} | {'Hit@1 (%)':<12} | {'Hit@3 (%)':<12} | {'MRR':<8}"
    print(header)
    print("-" * 80)

    best_config = None
    best_hit3 = -1
    best_mrr = -1

    for name, wd, ws, wg in test_configs:
        hit1, hit3, mrr_sum = 0, 0, 0.0

        for q in BENCHMARK_15_QUERIES:
            query = q["query"]
            exp_c = q["expected_chunks"]
            exp_e = q["expected_entities"]

            fused = {}
            c_map = {}

            # Dense search
            for rank, h in enumerate(retriever.dense.search(query, pool_k)):
                cid = h["chunk_index"]
                if cid < len(retriever.chunks):
                    k = retriever.chunks[cid]["chunk_id"]
                    c_map[k] = retriever.chunks[cid]
                    fused[k] = fused.get(k, 0.0) + wd / (rrf_k + rank + 1)

            # Sparse search
            for rank, h in enumerate(retriever.sparse.search(query, pool_k)):
                cid = h["chunk_index"]
                if cid < len(retriever.chunks):
                    k = retriever.chunks[cid]["chunk_id"]
                    c_map[k] = retriever.chunks[cid]
                    fused[k] = fused.get(k, 0.0) + ws / (rrf_k + rank + 1)

            # Graph search
            p_to_c = {c.get("place_id"): c for c in retriever.chunks if c.get("place_id")}
            for rank, g in enumerate(retriever.graph.search_subgraph(query, pool_k)):
                raw = g.get("chunk_id", "").replace("graph_", "")
                mc = p_to_c.get(raw)
                k = mc["chunk_id"] if mc else g["chunk_id"]
                if k not in c_map:
                    c_map[k] = mc if mc else g
                fused[k] = fused.get(k, 0.0) + wg / (rrf_k + rank + 1)

            ranked = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)[:3]

            q_rank = 0
            for rank_idx, k in enumerate(ranked, 1):
                chunk_obj = c_map[k]
                if is_match(chunk_obj, exp_c, exp_e):
                    q_rank = rank_idx
                    break

            if q_rank == 1:
                hit1 += 1
            if 1 <= q_rank <= 3:
                hit3 += 1
            if q_rank > 0:
                mrr_sum += 1.0 / q_rank

        h1_pct = (hit1 / 15.0) * 100
        h3_pct = (hit3 / 15.0) * 100
        mrr = mrr_sum / 15.0

        row = f"{name:<28} | {wd:<6.2f} {ws:<6.2f} {wg:<6.2f} | {h1_pct:>5.1f}% ({hit1:2d})  | {h3_pct:>5.1f}% ({hit3:2d})  | {mrr:.4f}"
        print(row)

        if h3_pct > best_hit3 or (h3_pct == best_hit3 and mrr > best_mrr):
            best_hit3 = h3_pct
            best_mrr = mrr
            best_config = (name, wd, ws, wg, h1_pct, h3_pct, mrr)

    print("=" * 80)
    print(f"🏆 BEST WEIGHT CONFIGURATION: {best_config[0]}")
    print(f"   Weights: Dense={best_config[1]}, Sparse={best_config[2]}, Graph={best_config[3]}")
    print(f"   Hit@1: {best_config[4]:.1f}%, Hit@3: {best_config[5]:.1f}%, MRR: {best_config[6]:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_weight_grid()
