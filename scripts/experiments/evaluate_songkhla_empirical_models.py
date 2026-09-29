import json
import pickle
import numpy as np
from pathlib import Path
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import sys

# Paths
BASE_DIR = Path('c:/social/A_krit2/finalproject')
DATA_DIR = BASE_DIR / 'data'

# Load Songkhla chunks and BM25
with open(DATA_DIR / 'songkhla_rag_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
with open(DATA_DIR / 'songkhla_bm25.pkl', 'rb') as f:
    bm25 = pickle.load(f)['bm25']
chunk_ids = [c['chunk_id'] for c in chunks]

# Load benchmark queries (from exp2, contains 20 queries)
sys.path.insert(0, str(BASE_DIR))
from scripts.experiments.exp2_retrieval_ablation import BENCHMARK_15_QUERIES as BENCHMARK_QUERIES

print(f"Total Songkhla Chunks: {len(chunks)}")
print(f"Total Benchmark Queries: {len(BENCHMARK_QUERIES)}")

# Models lineup
candidate_models = [
    ('WangchanBERTa-ConGen\n(Thai 768d)', 'kornwtp/ConGen-model-wangchanberta', False),
    ('MiniLM-L12\n(Multilingual 384d)', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', False),
    ('Multilingual-E5-Small\n(Multilingual 384d)', 'intfloat/multilingual-e5-small', True),
    ('DistilUSE-Base\n(Multilingual 512d)', 'sentence-transformers/distiluse-base-multilingual-cased-v2', False),
    ('Multilingual-E5-Base\n(Multilingual 768d)', 'intfloat/multilingual-e5-base', True),
    ('MPNet-Base\n(Multilingual 768d)', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', False)
]

results = []

for name, model_path, prefix in candidate_models:
    print(f"\nEvaluating: {name.replace(chr(10), ' ')}...")
    model = SentenceTransformer(model_path)
    
    corpus_texts = [f"passage: {c['content']}" if prefix else c['content'] for c in chunks]
    corpus_emb = model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    
    dense_ranks = []
    hybrid_ranks = []
    
    for q in BENCHMARK_QUERIES:
        q_text = f"query: {q['query']}" if prefix else q['query']
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        
        sims = np.dot(corpus_emb, q_emb)
        d_ids = [chunk_ids[i] for i in np.argsort(sims)[::-1]]
        
        bm_scores = bm25.get_scores(word_tokenize(q['query'], engine='newmm'))
        b_ids = [chunk_ids[i] for i in np.argsort(bm_scores)[::-1]]
        
        # Dual-RRF
        rrf = {}
        for r, cid in enumerate(d_ids): rrf[cid] = rrf.get(cid, 0) + 0.5 / (60 + r + 1)
        for r, cid in enumerate(b_ids): rrf[cid] = rrf.get(cid, 0) + 0.5 / (60 + r + 1)
        h_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
        
        expected_list = q.get('expected_chunks', [])
        
        # Rank is min rank among expected chunks
        d_r = 999
        h_r = 999
        for exp in expected_list:
            if exp in d_ids:
                d_r = min(d_r, d_ids.index(exp) + 1)
            if exp in h_ids:
                h_r = min(h_r, h_ids.index(exp) + 1)
                
        dense_ranks.append(d_r)
        hybrid_ranks.append(h_r)
        
    d_hit5 = np.mean([1 if r <= 5 else 0 for r in dense_ranks]) * 100
    h_hit5 = np.mean([1 if r <= 5 else 0 for r in hybrid_ranks]) * 100
    d_hit1 = np.mean([1 if r <= 1 else 0 for r in dense_ranks]) * 100
    h_hit1 = np.mean([1 if r <= 1 else 0 for r in hybrid_ranks]) * 100
    d_mrr = np.mean([1.0 / r if r < 900 else 0.0 for r in dense_ranks])
    h_mrr = np.mean([1.0 / r if r < 900 else 0.0 for r in hybrid_ranks])
    
    print(f"-> Dense Hit@5: {d_hit5:.1f}%, Hybrid Hit@5: {h_hit5:.1f}% (Diff: {h_hit5 - d_hit5:+.1f}%)")
    print(f"-> Dense Hit@1: {d_hit1:.1f}%, Hybrid Hit@1: {h_hit1:.1f}% | Dense MRR: {d_mrr:.4f}, Hybrid MRR: {h_mrr:.4f}")
    
    results.append({
        'name': name,
        'dense_hit5': d_hit5,
        'hybrid_hit5': h_hit5,
        'dense_hit1': d_hit1,
        'hybrid_hit1': h_hit1,
        'dense_mrr': d_mrr,
        'hybrid_mrr': h_mrr,
        'diff': h_hit5 - d_hit5
    })

with open(DATA_DIR / 'songkhla_empirical_models_benchmark.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nSaved benchmark results to songkhla_empirical_models_benchmark.json")
