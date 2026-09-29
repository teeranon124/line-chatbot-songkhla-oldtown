# -*- coding: utf-8 -*-
"""
Empirical & Unbiased Engineering Benchmark: Embedding Models for Songkhla Old Town RAG
=====================================================================================
Course: 241-351 AI for Social Media (Final Project)
Evaluates 3 Candidate Embedding Models:
  - Model A: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Compact Multilingual)
  - Model B: intfloat/multilingual-e5-base (Medium Multilingual with Asymmetric Prefix)
  - Model C: kornwtp/ConGen-model-wangchanberta (Thai Domain-Specific Sentence Transformer)

Transparent & Unbiased Scientific Metrics:
  1. Subword Token Fragmentation Rate (15 Domain-Specific Named Entities)
  2. Encoding Latency on CPU (10 Runs: Mean & Std Dev in ms)
  3. Retrieval Accuracy: Hit@1, Hit@3, MRR (Mean Reciprocal Rank), Cosine Margin
  4. Model Parameters & Memory Footprint
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Suppress TensorFlow warnings and avoid distutils conflicts in Python 3.12
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "songkhla_rag_chunks.json"
OUTPUT_JSON_PATH = DATA_DIR / "exp1_embedding_benchmark_results.json"

# 15 Domain-Specific Named Entities from Songkhla Old Town
DOMAIN_TERMS = [
    "หับ โห้ หิ้น",
    "เกียดฟั่ง",
    "แต้เฮี้ยงอิ้ว",
    "ซินกวนฮง",
    "บะเต็ง",
    "บ้านนครใน",
    "ศาลเจ้าพ่อหลักเมือง",
    "ไอติมโอ่ง",
    "ถนนนางงาม",
    "ชิโน-ยูโรเปียน",
    "ถนนนครนอก",
    "ถนนนครใน",
    "สตรีทอาร์ทสงขลา",
    "เขาตังกวน",
    "หาดสมิหลา"
]

# 15 In-Domain Evaluation Queries with Ground-Truth Target Chunks
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "type": "Simple Fact",
        "query": "ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่",
        "expected_chunk": "chunk_014"
    },
    {
        "id": 2,
        "type": "Simple Fact",
        "query": "ร้านเกียดฟั่งข้าวสตูเปิดปิดกี่โมง มีเมนูอะไรบ้าง",
        "expected_chunk": "chunk_013"
    },
    {
        "id": 3,
        "type": "Simple Fact",
        "query": "ร้านแต้เฮี้ยงอิ้วเปิดกี่รอบ และเบอร์โทรเบอร์อะไร",
        "expected_chunk": "chunk_015"
    },
    {
        "id": 4,
        "type": "Simple Fact",
        "query": "หอศิลป์สงขลาเปิดวันไหนบ้าง มีวันหยุดไหม",
        "expected_chunk": "chunk_004"
    },
    {
        "id": 5,
        "type": "Simple Fact",
        "query": "เบอร์โทรฉุกเฉินตำรวจท่องเที่ยวสงขลาคือเบอร์อะไร",
        "expected_chunk": "chunk_021"
    },
    {
        "id": 6,
        "type": "Multi-hop Relational",
        "query": "เดินอยู่ถนนนางงาม มีร้านอาหารและของหวานอะไรเปิดอยู่บ้าง",
        "expected_chunk": "chunk_014"
    },
    {
        "id": 7,
        "type": "Multi-hop Relational",
        "query": "พักที่โรงแรมสงขลาแต่แรก เดินไปสตรีทอาร์ทและโรงสีแดงกี่เมตร",
        "expected_chunk": "chunk_020"
    },
    {
        "id": 8,
        "type": "Multi-hop Relational",
        "query": "ตามแผนเที่ยววันที่ 1 มื้อเที่ยงกินที่ไหน และบ่ายโมงไปเที่ยวไหนต่อ",
        "expected_chunk": "chunk_001"
    },
    {
        "id": 9,
        "type": "Multi-hop Relational",
        "query": "มีงบประมาณ 100 บาท กินอะไรได้บ้างในย่านเมืองเก่าสงขลา",
        "expected_chunk": "chunk_014"
    },
    {
        "id": 10,
        "type": "Multi-hop Relational",
        "query": "รถรางชมเมืองสงขลาขึ้นที่ไหน และพาไปชมจุดสำคัญอะไรบ้าง",
        "expected_chunk": "chunk_011"
    },
    {
        "id": 11,
        "type": "Complex Synthesis",
        "query": "เล่าประวัติความเป็นมาของโรงสีแดงหับโห้หิ้น ยุค ร.6 ให้ฟังหน่อย",
        "expected_chunk": "chunk_006"
    },
    {
        "id": 12,
        "type": "Complex Synthesis",
        "query": "บ้านนครในมีความสำคัญทางประวัติศาสตร์อย่างไร และมีของสะสมอะไร",
        "expected_chunk": "chunk_005"
    },
    {
        "id": 13,
        "type": "Complex Synthesis",
        "query": "จุดกำเนิดของ 3 ถนนสายวัฒนธรรมในเมืองเก่าสงขลาเริ่มต้นอย่างไร",
        "expected_chunk": "chunk_003"
    },
    {
        "id": 14,
        "type": "Complex Synthesis",
        "query": "ช่วยวางแผนเที่ยวสงขลา 1 วันเต็ม สำหรับคนที่อยากเน้นกินของอร่อยและถ่ายรูปตึกเก่า",
        "expected_chunk": "chunk_001"
    },
    {
        "id": 15,
        "type": "Complex Synthesis",
        "query": "เปรียบเทียบจุดเด่นของโรงแรมสงขลาแต่แรก กับ โรงแรมคลับทรี",
        "expected_chunk": "chunk_019"
    }
]

MODEL_SPECS = [
    {
        "id": "model_a",
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_path": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "category": "Multilingual Small",
        "use_e5_prefix": False
    },
    {
        "id": "model_b_raw",
        "name": "multilingual-e5-base (Raw)",
        "hf_path": "intfloat/multilingual-e5-base",
        "category": "Multilingual Medium (Raw)",
        "use_e5_prefix": False
    },
    {
        "id": "model_b_prefix",
        "name": "multilingual-e5-base (Prefix)",
        "hf_path": "intfloat/multilingual-e5-base",
        "category": "Multilingual Medium (E5-Instructed)",
        "use_e5_prefix": True
    },
    {
        "id": "model_c",
        "name": "ConGen-WangchanBERTa",
        "hf_path": "kornwtp/ConGen-model-wangchanberta",
        "category": "Thai Monolingual Domain",
        "use_e5_prefix": False
    }
]


def load_chunks() -> List[Dict[str, Any]]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[*] Loaded {len(chunks)} knowledge chunks from {CHUNKS_PATH.name}")
    return chunks


def evaluate_tokenization(model, term_list: List[str]) -> Dict[str, Any]:
    tokenizer = model.tokenizer
    vocab_size = len(tokenizer)
    term_details = []
    token_counts = []
    
    for term in term_list:
        tokens = tokenizer.tokenize(term)
        token_count = len(tokens)
        token_counts.append(token_count)
        term_details.append({
            "term": term,
            "char_length": len(term),
            "token_count": token_count,
            "tokens": tokens,
            "fragmentation_ratio": round(token_count / max(1, len(term.split())), 2)
        })
    
    return {
        "vocab_size": vocab_size,
        "total_tokens_for_15_terms": int(np.sum(token_counts)),
        "mean_tokens_per_term": float(np.mean(token_counts)),
        "std_tokens_per_term": float(np.std(token_counts)),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "term_details": term_details
    }


def evaluate_latency(model, sample_sentences: List[str], iterations: int = 10) -> Dict[str, Any]:
    # Warm-up runs (2 times)
    for _ in range(2):
        _ = model.encode(sample_sentences, normalize_embeddings=True, show_progress_bar=False)
    
    run_latencies_ms = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model.encode(sample_sentences, normalize_embeddings=True, show_progress_bar=False)
        t1 = time.perf_counter()
        run_latencies_ms.append((t1 - t0) * 1000.0)
    
    return {
        "iterations": iterations,
        "batch_size": len(sample_sentences),
        "mean_latency_ms": float(np.mean(run_latencies_ms)),
        "std_latency_ms": float(np.std(run_latencies_ms)),
        "min_latency_ms": float(np.min(run_latencies_ms)),
        "max_latency_ms": float(np.max(run_latencies_ms)),
        "ms_per_sentence": float(np.mean(run_latencies_ms) / len(sample_sentences)),
        "throughput_sentences_per_sec": float(len(sample_sentences) / (np.mean(run_latencies_ms) / 1000.0))
    }


def evaluate_retrieval(model, chunks: List[Dict[str, Any]], queries: List[Dict[str, Any]], use_e5_prefix: bool = False) -> Dict[str, Any]:
    # Prepare chunk texts
    chunk_ids = [c["chunk_id"] for c in chunks]
    if use_e5_prefix:
        chunk_texts = [f"passage: {c['title']} {c['content']}" for c in chunks]
    else:
        chunk_texts = [f"{c['title']} {c['content']}" for c in chunks]
    
    # Encode all chunks into matrix: shape (N_chunks, dim)
    chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True, show_progress_bar=False)
    chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
    
    query_evaluations = []
    reciprocal_ranks = []
    hit_1_count = 0
    hit_3_count = 0
    positive_sims = []
    top_negative_sims = []
    margins = []
    
    for q_item in queries:
        q_text = q_item["query"]
        expected_chunk_id = q_item["expected_chunk"]
        
        if use_e5_prefix:
            encoded_q = model.encode(f"query: {q_text}", normalize_embeddings=True, show_progress_bar=False)
        else:
            encoded_q = model.encode(q_text, normalize_embeddings=True, show_progress_bar=False)
            
        encoded_q = np.array(encoded_q, dtype=np.float32)
        
        # Cosine similarity is dot product because vectors are normalized
        sim_scores = np.dot(chunk_embeddings, encoded_q)
        ranked_indices = np.argsort(-sim_scores)
        
        # Find position of expected chunk
        expected_idx = chunk_ids.index(expected_chunk_id)
        rank = int(np.where(ranked_indices == expected_idx)[0][0]) + 1
        rr = 1.0 / rank
        reciprocal_ranks.append(rr)
        
        pos_sim = float(sim_scores[expected_idx])
        positive_sims.append(pos_sim)
        
        # Best negative
        top_neg_sim = -1.0
        for idx in ranked_indices:
            if idx != expected_idx:
                top_neg_sim = float(sim_scores[idx])
                break
        top_negative_sims.append(top_neg_sim)
        margin = pos_sim - top_neg_sim
        margins.append(margin)
        
        is_hit_1 = (rank == 1)
        is_hit_3 = (rank <= 3)
        if is_hit_1:
            hit_1_count += 1
        if is_hit_3:
            hit_3_count += 1
            
        query_evaluations.append({
            "query_id": q_item["id"],
            "query": q_text,
            "type": q_item["type"],
            "expected_chunk": expected_chunk_id,
            "retrieved_rank_1": chunk_ids[ranked_indices[0]],
            "ground_truth_rank": rank,
            "hit_at_1": bool(is_hit_1),
            "hit_at_3": bool(is_hit_3),
            "reciprocal_rank": round(rr, 4),
            "positive_similarity": round(pos_sim, 4),
            "top_negative_similarity": round(top_neg_sim, 4),
            "cosine_margin": round(margin, 4)
        })
        
    num_q = len(queries)
    return {
        "mrr": float(np.mean(reciprocal_ranks)),
        "hit_at_1_accuracy": float(hit_1_count / num_q),
        "hit_at_3_accuracy": float(hit_3_count / num_q),
        "mean_positive_similarity": float(np.mean(positive_sims)),
        "mean_top_negative_similarity": float(np.mean(top_negative_sims)),
        "mean_cosine_margin": float(np.mean(margins)),
        "query_evaluations": query_evaluations
    }


def main():
    print("=" * 80)
    print("  EMPIRICAL BENCHMARK: EMBEDDING MODELS FOR SONGKHLA OLD TOWN RAG")
    print("  Comparing MiniLM-L12 vs E5-Base vs WangchanBERTa")
    print("=" * 80)
    
    from sentence_transformers import SentenceTransformer
    
    chunks = load_chunks()
    
    # 15 representative sample sentences for latency evaluation
    sample_sentences = [q["query"] for q in BENCHMARK_QUERIES]
    
    all_benchmark_results = {
        "benchmark_metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_chunks": len(chunks),
            "num_test_queries": len(BENCHMARK_QUERIES),
            "num_domain_terms": len(DOMAIN_TERMS),
            "device": "CPU",
            "eval_framework": "sentence-transformers / PyTorch"
        },
        "models": {}
    }
    
    loaded_models_cache = {}
    
    for spec in MODEL_SPECS:
        m_id = spec["id"]
        m_name = spec["name"]
        hf_path = spec["hf_path"]
        category = spec["category"]
        use_prefix = spec["use_e5_prefix"]
        
        print(f"\n[{m_id.upper()}] Loading model: {hf_path} ({category})...")
        if hf_path in loaded_models_cache:
            model = loaded_models_cache[hf_path]
        else:
            model = SentenceTransformer(hf_path)
            loaded_models_cache[hf_path] = model
            
        # Get parameter count and embedding dimension
        param_count = sum(p.numel() for p in model.parameters())
        emb_dim = model.get_sentence_embedding_dimension()
        print(f"    - Parameters: {param_count:,} ({param_count/1e6:.1f}M)")
        print(f"    - Embedding Dimension: {emb_dim}")
        
        # 1. Tokenization evaluation
        print(f"    - Running Subword Token Fragmentation evaluation on 15 Songkhla terms...")
        tok_results = evaluate_tokenization(model, DOMAIN_TERMS)
        print(f"      Mean tokens/term: {tok_results['mean_tokens_per_term']:.2f} (Total: {tok_results['total_tokens_for_15_terms']})")
        
        # 2. Latency evaluation
        print(f"    - Measuring CPU Latency over 10 iterations (batch of {len(sample_sentences)} sentences)...")
        lat_results = evaluate_latency(model, sample_sentences, iterations=10)
        print(f"      Mean Latency: {lat_results['mean_latency_ms']:.2f} +/- {lat_results['std_latency_ms']:.2f} ms")
        print(f"      Throughput: {lat_results['throughput_sentences_per_sec']:.1f} sent/sec")
        
        # 3. Retrieval MRR & Cosine evaluation
        print(f"    - Measuring In-Domain Retrieval (Hit@1, Hit@3, MRR, Cosine Margin)...")
        ret_results = evaluate_retrieval(model, chunks, BENCHMARK_QUERIES, use_e5_prefix=use_prefix)
        print(f"      Hit@1: {ret_results['hit_at_1_accuracy']*100:.1f}% | Hit@3: {ret_results['hit_at_3_accuracy']*100:.1f}% | MRR: {ret_results['mrr']:.4f}")
        print(f"      Mean Cosine Margin: {ret_results['mean_cosine_margin']:.4f}")
        
        all_benchmark_results["models"][m_id] = {
            "model_name": m_name,
            "hf_path": hf_path,
            "category": category,
            "use_e5_prefix": use_prefix,
            "param_count_m": round(param_count / 1e6, 2),
            "embedding_dimension": emb_dim,
            "tokenization": tok_results,
            "latency_cpu_ms": lat_results,
            "retrieval": ret_results
        }

    # Save to JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_results, f, ensure_ascii=False, indent=2)
    print(f"\n[+] Full Benchmark Data successfully exported to {OUTPUT_JSON_PATH}")

    # Print Clean Comparison Table
    print("\n" + "=" * 95)
    print(f"{'Model':<35} | {'Params':<7} | {'Dim':<5} | {'Tokens/Term':<12} | {'Latency(ms)':<14} | {'Hit@1':<7} | {'MRR':<7}")
    print("-" * 95)
    for m_id, data in all_benchmark_results["models"].items():
        name = data["model_name"]
        params = f"{data['param_count_m']}M"
        dim = f"{data['embedding_dimension']}"
        tok_avg = f"{data['tokenization']['mean_tokens_per_term']:.2f}"
        lat = f"{data['latency_cpu_ms']['mean_latency_ms']:.1f}±{data['latency_cpu_ms']['std_latency_ms']:.1f}"
        hit1 = f"{data['retrieval']['hit_at_1_accuracy']*100:.1f}%"
        mrr = f"{data['retrieval']['mrr']:.3f}"
        print(f"{name:<35} | {params:<7} | {dim:<5} | {tok_avg:<12} | {lat:<14} | {hit1:<7} | {mrr:<7}")
    print("=" * 95)


if __name__ == "__main__":
    main()
