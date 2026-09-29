import json
import numpy as np
from pathlib import Path
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import pickle
import sys

DATA_DIR = Path('c:/social/A_krit2/finalproject/data')
with open(DATA_DIR / 'songkhla_rag_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)
with open(DATA_DIR / 'songkhla_bm25.pkl', 'rb') as f:
    bm25 = pickle.load(f)['bm25']
chunk_ids = [c['chunk_id'] for c in chunks]

sys.path.insert(0, 'c:/social/A_krit2/finalproject')
from scripts.experiments.exp1_embedding_benchmark import BENCHMARK_QUERIES as queries

models = [
    ('MiniLM-L12\n(Multilingual 384d)', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', False),
    ('WangchanBERTa-ConGen\n(Thai 768d)', 'kornwtp/ConGen-model-wangchanberta', False),
    ('Multilingual-E5-Base\n(Multilingual 768d)', 'intfloat/multilingual-e5-base', True)
]

print("Calculating Hit@1, Hit@3, Hit@5...")
for name, path, prefix in models:
    model = SentenceTransformer(path)
    corpus_texts = [f"passage: {c['content']}" if prefix else c['content'] for c in chunks]
    corpus_emb = model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
    
    dense_ranks = []
    hybrid_ranks = []
    for q in queries:
        q_text = f"query: {q['query']}" if prefix else q['query']
        q_emb = model.encode([q_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = np.dot(corpus_emb, q_emb)
        d_ids = [chunk_ids[i] for i in np.argsort(sims)[::-1]]
        
        bm_scores = bm25.get_scores(word_tokenize(q['query'], engine='newmm'))
        b_ids = [chunk_ids[i] for i in np.argsort(bm_scores)[::-1]]
        
        rrf = {}
        for r, cid in enumerate(d_ids): rrf[cid] = rrf.get(cid, 0) + 0.5/(60+r+1)
        for r, cid in enumerate(b_ids): rrf[cid] = rrf.get(cid, 0) + 0.5/(60+r+1)
        h_ids = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)
        
        expected = q['expected_chunk']
        dense_ranks.append(d_ids.index(expected)+1 if expected in d_ids else 99)
        hybrid_ranks.append(h_ids.index(expected)+1 if expected in h_ids else 99)
        
    for k in [1, 3, 5]:
        d_hit = np.mean([1 if r <= k else 0 for r in dense_ranks])*100
        h_hit = np.mean([1 if r <= k else 0 for r in hybrid_ranks])*100
        print(f"{name.replace(chr(10), ' ')} | Hit@{k}: Dense={d_hit:.1f}% -> Hybrid={h_hit:.1f}% (Diff: {h_hit-d_hit:+.1f}%)")
    print("-" * 50)
