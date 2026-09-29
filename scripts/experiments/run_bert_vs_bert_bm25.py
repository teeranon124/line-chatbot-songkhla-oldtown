import os
import sys
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\social\A_krit2\finalproject")
DATA_DIR = BASE_DIR / "data"

with open(DATA_DIR / "songkhla_rag_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

from pythainlp.tokenize import word_tokenize
import pickle
with open(DATA_DIR / "songkhla_bm25.pkl", "rb") as f:
    bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
chunk_ids = [c["chunk_id"] for c in chunks]

sys.path.insert(0, str(BASE_DIR))
from scripts.experiments.exp1_embedding_benchmark import BENCHMARK_QUERIES as queries

from sentence_transformers import SentenceTransformer

models_to_test = [
    {
        "name": "MiniLM-L12",
        "path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "prefix": False
    },
    {
        "name": "mE5-Base",
        "path": "intfloat/multilingual-e5-base",
        "prefix": True
    },
    {
        "name": "ConGen WangchanBERTa",
        "path": "kornwtp/ConGen-model-wangchanberta",
        "prefix": False
    }
]

print("Running BERT vs BERT + BM25 Shootout...")
results = []

for m_info in models_to_test:
    print(f"Loading {m_info['name']}...")
    try:
        model = SentenceTransformer(m_info["path"])
    except Exception as e:
        print(f"Error loading {m_info['name']}: {e}")
        continue

    # Encode corpus
    corpus_texts = [f"passage: {c['content']}" if m_info["prefix"] else c["content"] for c in chunks]
    corpus_emb = model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)

    dense_ranks = []
    hybrid_ranks = []

    for q in queries:
        query_text = q["query"]
        expected_chunk = q["expected_chunk"]
        
        # 1. Dense Search
        q_emb_text = f"query: {query_text}" if m_info["prefix"] else query_text
        q_emb = model.encode([q_emb_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        
        sims = np.dot(corpus_emb, q_emb)
        dense_sorted_indices = np.argsort(sims)[::-1]
        dense_sorted_ids = [chunk_ids[idx] for idx in dense_sorted_indices]

        # 2. BM25 Search
        q_tokens = word_tokenize(query_text, engine="newmm")
        bm25_scores = bm25.get_scores(q_tokens)
        bm25_sorted_indices = np.argsort(bm25_scores)[::-1]
        bm25_sorted_ids = [chunk_ids[idx] for idx in bm25_sorted_indices]

        # 3. Hybrid RRF (k=60)
        rrf_scores = {}
        for rank, cid in enumerate(dense_sorted_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 0.50 / (60 + rank + 1)
        for rank, cid in enumerate(bm25_sorted_ids):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 0.50 / (60 + rank + 1)

        hybrid_sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Record ranks
        d_rank = dense_sorted_ids.index(expected_chunk) + 1 if expected_chunk in dense_sorted_ids else 999
        h_rank = hybrid_sorted_ids.index(expected_chunk) + 1 if expected_chunk in hybrid_sorted_ids else 999
        
        dense_ranks.append(d_rank)
        hybrid_ranks.append(h_rank)

    # Compute Metrics
    dense_hit1 = np.mean([1 if r == 1 else 0 for r in dense_ranks])
    dense_hit3 = np.mean([1 if r <= 3 else 0 for r in dense_ranks])
    dense_mrr = np.mean([1.0 / r for r in dense_ranks])

    hybrid_hit1 = np.mean([1 if r == 1 else 0 for r in hybrid_ranks])
    hybrid_hit3 = np.mean([1 if r <= 3 else 0 for r in hybrid_ranks])
    hybrid_mrr = np.mean([1.0 / r for r in hybrid_ranks])

    results.append({
        "model": m_info["name"],
        "dense_hit1": dense_hit1,
        "dense_hit3": dense_hit3,
        "dense_mrr": dense_mrr,
        "hybrid_hit1": hybrid_hit1,
        "hybrid_hit3": hybrid_hit3,
        "hybrid_mrr": hybrid_mrr,
        "delta_mrr": hybrid_mrr - dense_mrr
    })

print("\n=== FINAL RESULTS: BERT vs BERT + BM25 ===")
for r in results:
    print(f"Model: {r['model']}")
    print(f"  Dense Only:   Hit@1={r['dense_hit1']*100:.1f}%, Hit@3={r['dense_hit3']*100:.1f}%, MRR={r['dense_mrr']:.4f}")
    print(f"  BERT + BM25:  Hit@1={r['hybrid_hit1']*100:.1f}%, Hit@3={r['hybrid_hit3']*100:.1f}%, MRR={r['hybrid_mrr']:.4f}")
    print(f"  Improvement:  Delta MRR = +{r['delta_mrr']:.4f}")
    print()

with open(DATA_DIR / "exp1_bert_vs_bert_bm25_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Saved to exp1_bert_vs_bert_bm25_results.json")
