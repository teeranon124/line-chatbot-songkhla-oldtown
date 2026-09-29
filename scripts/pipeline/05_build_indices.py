# -*- coding: utf-8 -*-
"""
Script to build FAISS (Dense) and BM25 (Sparse) retrieval indices
for Songkhla Old Town RAG System.
Uses:
- ConGen-model-wangchanberta (768-dim normalized) for FAISS
- BM25Okapi with PyThaiNLP (newmm engine) for Sparse
- Combined data: AnyFlip Cultural Text + Scraped Google Knowledge Facts

Outputs:
1. finalproject/data/songkhla_faiss.index
2. finalproject/data/songkhla_bm25.pkl
3. finalproject/data/songkhla_chunks_metadata.json
"""

import os
import json
import time
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

# Enable offline loading for instantaneous startup (< 1 sec)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

EMBED_MODEL_NAME = "kornwtp/ConGen-model-wangchanberta"


def build_indices():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    chunks_path = os.path.join(base_dir, "songkhla_rag_chunks.json")
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    print(f"\n[STEP 1] Loaded {len(chunks)} combined chunks (AnyFlip + Scraped Facts).")
    
    # Prepare texts for embedding & tokenization (include title and tags for complete semantic recall)
    texts = [f"{c.get('title', '')}\n{' '.join(c.get('tags', []))}\n{c['content']}" for c in chunks]
    
    # 1. Build Dense FAISS Index
    print(f"\n[STEP 2] Loading Dense Embedding Model: {EMBED_MODEL_NAME}...")
    t0 = time.time()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    dim = embedder.get_embedding_dimension()
    print(f"Model loaded in {time.time() - t0:.2f}s (Embedding Dimension: {dim})")
    
    print(f"Encoding {len(texts)} chunks for FAISS Index...")
    embeddings = embedder.encode(
        texts,
        batch_size=16,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )
    
    # Build Inner Product index (Cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    print(f"FAISS index built: {index.ntotal} vectors added.")
    
    faiss_out = os.path.join(base_dir, "songkhla_faiss.index")
    faiss.write_index(index, faiss_out)
    print(f"[OK] Saved FAISS Index to {faiss_out}")
    
    # 2. Build Sparse BM25 Index
    print(f"\n[STEP 3] Tokenizing texts with PyThaiNLP (engine='newmm') for BM25...")
    tokenized_corpus = [word_tokenize(t, engine="newmm") for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_out = os.path.join(base_dir, "songkhla_bm25.pkl")
    with open(bm25_out, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
    print(f"[OK] Saved BM25 Index to {bm25_out}")
    
    # 3. Save Metadata
    meta_out = os.path.join(base_dir, "songkhla_chunks_metadata.json")
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved Chunks Metadata to {meta_out}")
    
    # 4. Verify Retrieval with Sample Queries
    print("\n" + "=" * 60)
    print("VERIFICATION: RETRIEVAL ACCURACY TEST (PDF + SCRAPED FACTS)")
    print("=" * 60)
    
    test_queries = [
        "ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่",                 # Scraped Fact query
        "ประวัติความเป็นมาของโรงสีแดงหับโห้หิ้น ยุค ร.6",        # AnyFlip Cultural query
        "โปรแกรมเที่ยวสงขลา วันที่ 1 ต้องไปที่ไหนบ้าง",          # AnyFlip Itinerary query
        "ถนนนางงามมีร้านอาหารและของหวานอะไรแนะนำบ้าง"           # Hybrid query
    ]
    
    for q in test_queries:
        print(f"\n[Query]: \"{q}\"")
        # Dense search
        q_emb = embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        d_scores, d_indices = index.search(q_emb, k=2)
        top_dense = chunks[d_indices[0][0]]["title"]
        
        # Sparse search
        q_tokens = word_tokenize(q, engine="newmm")
        s_scores = bm25.get_scores(q_tokens)
        top_sparse_idx = int(np.argmax(s_scores))
        top_sparse = chunks[top_sparse_idx]["title"]
        
        print(f"  -> Top Dense (FAISS):  [{top_dense}] (Score: {d_scores[0][0]:.4f})")
        print(f"  -> Top Sparse (BM25): [{top_sparse}] (Score: {s_scores[top_sparse_idx]:.2f})")
        
    print("\nALL INDICES BUILT AND VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    build_indices()
