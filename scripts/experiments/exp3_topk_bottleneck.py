# -*- coding: utf-8 -*-
"""
================================================================================
EXPERIMENT 3: Context Bottleneck Experiment (Top-K Limitation & Dynamic Top-K)
Course: 241-351 AI for Social Media (Final Project)
Student: Teeranon Thongseedam (6710110196)
================================================================================
This engineering experiment rigorously evaluates the Context Bottleneck phenomenon
in RAG systems across 4 Context Retrieval Configurations:
  1. Fixed k=2
  2. Fixed k=4 (Industry Standard Baseline)
  3. Fixed k=7
  4. Dynamic Top-K (Adaptive intent-driven: Group A -> 2, Group B -> 4, Group C -> 7)

Tested across 3 In-Domain Query Groups:
  - Group A: Simple Fact-seeking (e.g. ร้านแต้เปิดกี่โมง)
  - Group B: Comparison / Area info (e.g. แนะนำของกินถนนนางงาม)
  - Group C: Complex Multi-constraint (e.g. จัดตารางเที่ยว 2 วัน พร้อมคำนวณงบประมาณและเส้นทางเดินเท้าเชื่อม 3 ถนน)

Objective Evaluation Metrics:
  1. Target Context Retention Rate (% of ground-truth chunks retrieved)
  2. Budget/Price Information Retention Rate (% of price-bearing chunks retrieved)
  3. Prompt Context Token Overhead (Exact LLM BPE tokens via Ollama prompt_eval_count & PyThaiNLP words)
  4. Real LLM Generation & Empirical Omission Rate (Testing with Ollama qwen2.5:3b)
  5. Retrieval & Generation Latency (seconds)
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import requests
from pythainlp.tokenize import word_tokenize

# Set offline huggingface flags
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import HybridRetriever

# -----------------------------------------------------------------------------
# BENCHMARK TEST SUITE (9 Queries across 3 Groups)
# -----------------------------------------------------------------------------
BENCHMARK_SUITE = [
    # Group A: Simple Fact
    {
        "id": "A1",
        "group": "Group A: Simple Fact",
        "intent": "Simple Fact",
        "query": "ร้านแต้เฮี้ยงอิ้วเปิดกี่โมงและมีเบอร์โทรอะไร",
        "target_chunks": ["chunk_015"],
        "budget_chunks": ["chunk_015"],
        "dynamic_k": 2
    },
    {
        "id": "A2",
        "group": "Group A: Simple Fact",
        "intent": "Simple Fact",
        "query": "ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่",
        "target_chunks": ["chunk_014"],
        "budget_chunks": ["chunk_014"],
        "dynamic_k": 2
    },
    {
        "id": "A3",
        "group": "Group A: Simple Fact",
        "intent": "Simple Fact",
        "query": "หอศิลป์สงขลาเปิดวันไหนบ้าง มีวันหยุดไหม",
        "target_chunks": ["chunk_004"],
        "budget_chunks": [],
        "dynamic_k": 2
    },

    # Group B: Comparison / Area info
    {
        "id": "B1",
        "group": "Group B: Comparison / Area info",
        "intent": "Comparison / Area info",
        "query": "แนะนำของกินถนนนางงาม มีร้านเด็ดอะไรบ้างและราคาประมาณเท่าไหร่",
        "target_chunks": ["chunk_013", "chunk_014", "chunk_015", "chunk_018"],
        "budget_chunks": ["chunk_013", "chunk_014", "chunk_015", "chunk_018"],
        "dynamic_k": 4
    },
    {
        "id": "B2",
        "group": "Group B: Comparison / Area info",
        "intent": "Comparison / Area info",
        "query": "เปรียบเทียบจุดเด่นและราคาโรงแรมในย่านเมืองเก่าสงขลา เช่น โรงแรมสงขลาแต่แรก กับ Club Tree",
        "target_chunks": ["chunk_019", "chunk_020"],
        "budget_chunks": ["chunk_019"],
        "dynamic_k": 4
    },
    {
        "id": "B3",
        "group": "Group B: Comparison / Area info",
        "intent": "Comparison / Area info",
        "query": "เดินเล่นถนนนครนอก มีสถานที่สำคัญและของกินอะไรบ้าง",
        "target_chunks": ["chunk_004", "chunk_005", "chunk_006", "chunk_007"],
        "budget_chunks": [],
        "dynamic_k": 4
    },

    # Group C: Complex Multi-constraint
    {
        "id": "C1",
        "group": "Group C: Complex Multi-constraint",
        "intent": "Complex Multi-constraint",
        "query": "จัดตารางเที่ยว 2 วัน พร้อมคำนวณงบประมาณและเส้นทางเดินเท้าเชื่อม 3 ถนน",
        "target_chunks": ["chunk_001", "chunk_002", "chunk_003", "chunk_014", "chunk_015", "chunk_019", "chunk_020"],
        "budget_chunks": ["chunk_014", "chunk_015", "chunk_019"],
        "dynamic_k": 7
    },
    {
        "id": "C2",
        "group": "Group C: Complex Multi-constraint",
        "intent": "Complex Multi-constraint",
        "query": "วางแผนเที่ยวสงขลา 2 วัน 1 คืน คำนวณงบประมาณค่าอาหารร้านแต้ ไอติมโอ่ง และค่าที่พักโรงแรมสงขลาแต่แรก",
        "target_chunks": ["chunk_001", "chunk_002", "chunk_014", "chunk_015", "chunk_019", "chunk_020"],
        "budget_chunks": ["chunk_014", "chunk_015", "chunk_019"],
        "dynamic_k": 7
    },
    {
        "id": "C3",
        "group": "Group C: Complex Multi-constraint",
        "intent": "Complex Multi-constraint",
        "query": "จัดทริปเดินเท้า 3 ถนนสายวัฒนธรรมพร้อมคำนวณงบประมาณค่ากินของคาวของหวาน (เกียดฟั่ง ไอติมโอ่ง แต้เฮี้ยงอิ้ว) และของฝาก",
        "target_chunks": ["chunk_003", "chunk_013", "chunk_014", "chunk_015", "chunk_018", "chunk_020"],
        "budget_chunks": ["chunk_013", "chunk_014", "chunk_015", "chunk_018"],
        "dynamic_k": 7
    }
]

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """คุณคือ "น้องสิงขร" ผู้ช่วยอัจฉริยะนำเที่ยวย่านเมืองเก่าสงขลา
หน้าที่ของคุณ:
1. ตอบให้ 'สั้น กระชับ ตรงประเด็น' กับคำถามที่สุด
2. ห้ามมีคำเกริ่นทักทาย ให้ตอบเข้าเนื้อหาทันที
3. อ้างอิงข้อมูลจากบริบทเท่านั้น หากไม่มีข้อมูลให้ระบุอย่างชัดเจนว่า "ไม่มีข้อมูลในเอกสาร" หรือ "ไม่สามารถคำนวณได้เนื่องจากไม่มีข้อมูลราคาในเอกสาร"
4. ห้ามแต่งเติมหรือคาดเดาตัวเลขราคาและงบประมาณขึ้นมาเองเด็ดขาด"""


def build_context_string(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", f"เอกสารที่ {i}")
        content = c.get("content", "").strip()
        parts.append(f"--- [เอกสารที่ {i}: {title}] ---\n{content}\n")
    return "\n".join(parts)


def call_ollama_llm(prompt: str, model_name: str = "qwen2.5:3b") -> Dict[str, Any]:
    """Invokes local Ollama and collects generation metrics."""
    t0 = time.time()
    try:
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "top_p": 0.9}
            },
            timeout=75
        )
        latency = time.time() - t0
        if res.status_code == 200:
            data = res.json()
            return {
                "answer": data.get("message", {}).get("content", "").strip(),
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "latency_sec": round(latency, 2),
                "error": None
            }
        return {
            "answer": f"Ollama HTTP {res.status_code}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_sec": round(latency, 2),
            "error": res.text
        }
    except Exception as e:
        return {
            "answer": f"Ollama Exception: {e}",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "latency_sec": round(time.time() - t0, 2),
            "error": str(e)
        }


def detect_budget_omission(answer: str, budget_chunks_count: int) -> bool:
    """
    Empirically detects whether the LLM answer suffers from budget omission
    (i.e. Unable to answer budget/price or explicitly states data is missing in document).
    """
    if budget_chunks_count == 0:
        return False
    
    omission_phrases = [
        "ไม่มีข้อมูล", "ไม่พบข้อมูล", "ไม่มีรายละเอียด",
        "ไม่สามารถคำนวณ", "ไม่สามารถระบุ", "ไม่ได้ระบุราคา",
        "ไม่มีราคา", "ต้องหาข้อมูลเพิ่มเติม"
    ]
    # Check if budget/price was asked and answered with omission phrases
    has_omission_phrase = any(phrase in answer for phrase in omission_phrases)
    
    # Check if numbers with "บาท" exist
    has_price_number = "บาท" in answer
    
    # If the model explicitly states missing info regarding price/budget
    if ("ไม่มีข้อมูล" in answer or "ไม่สามารถคำนวณ" in answer or "ไม่มีรายละเอียด" in answer) and "บาท" not in answer:
        return True
    
    # If it has omission phrase specifically attached to cost/budget
    for p in ["ค่าอาหาร", "ราคา", "งบประมาณ", "ค่าที่พัก"]:
        for om in ["ไม่มีข้อมูล", "ไม่สามารถคำนวณ", "ไม่มีรายละเอียด"]:
            if f"{p}{om}" in answer or f"{p} {om}" in answer or f"{p}: {om}" in answer:
                return True
                
    return False


# -----------------------------------------------------------------------------
# MAIN EXPERIMENT EXECUTION
# -----------------------------------------------------------------------------
def run_topk_bottleneck_experiment():
    print("=" * 80)
    print("🔬 EXPERIMENT 3: CONTEXT BOTTLENECK EXPERIMENT (TOP-K LIMITATION)")
    print("   Benchmarking Fixed k=2, Fixed k=4, Fixed k=7 vs Dynamic Top-K")
    print("=" * 80)

    retriever = HybridRetriever()

    strategies = [
        {"name": "Fixed k=2", "k_type": "fixed", "val": 2},
        {"name": "Fixed k=4 (Baseline)", "k_type": "fixed", "val": 4},
        {"name": "Fixed k=7", "k_type": "fixed", "val": 7},
        {"name": "Dynamic Top-K", "k_type": "dynamic", "val": None}
    ]

    all_raw_results = []
    summary_by_strategy_group = {}

    for strat in strategies:
        strat_name = strat["name"]
        print(f"\n=======================================================")
        print(f"▶ TESTING STRATEGY: {strat_name}")
        print(f"=======================================================")

        strat_results = []

        for item in BENCHMARK_SUITE:
            qid = item["id"]
            group = item["group"]
            query = item["query"]
            target_chunks = item["target_chunks"]
            budget_chunks = item["budget_chunks"]

            # Determine Top-K
            if strat["k_type"] == "fixed":
                k = strat["val"]
            else:
                k = item["dynamic_k"]

            # 1. Retrieve Context
            t_ret0 = time.time()
            retrieved_docs = retriever.retrieve(query, top_k=k, mode="hybrid")
            ret_latency = round(time.time() - t_ret0, 4)

            retrieved_cids = [c["chunk_id"] for c in retrieved_docs]

            # 2. Context Metrics
            context_str = build_context_string(retrieved_docs)
            pythai_words = len(word_tokenize(context_str, engine="newmm"))
            char_count = len(context_str)

            # Target Retention
            hit_targets = set(retrieved_cids).intersection(set(target_chunks))
            target_retention_pct = (len(hit_targets) / len(target_chunks)) * 100.0 if target_chunks else 100.0

            # Budget Retention
            hit_budget = set(retrieved_cids).intersection(set(budget_chunks))
            budget_retention_pct = (len(hit_budget) / len(budget_chunks)) * 100.0 if budget_chunks else 100.0

            # 3. LLM Generation
            user_prompt = (
                f"ข้อมูลบริบทอ้างอิง:\n{context_str}\n\n"
                f"คำถามของนักท่องเที่ยว: {query}\n"
                f"คำตอบของน้องสิงขร:"
            )

            llm_res = call_ollama_llm(user_prompt, model_name="qwen2.5:3b")
            answer = llm_res["answer"]
            prompt_tokens = llm_res["prompt_tokens"]
            completion_tokens = llm_res["completion_tokens"]
            llm_latency = llm_res["latency_sec"]

            # 4. Omission Detection
            is_omitted = detect_budget_omission(answer, len(budget_chunks))

            record = {
                "strategy": strat_name,
                "query_id": qid,
                "group": group,
                "intent": item["intent"],
                "k": k,
                "query": query,
                "retrieved_chunks": retrieved_cids,
                "target_chunks": target_chunks,
                "hit_targets": list(hit_targets),
                "target_retention_pct": round(target_retention_pct, 1),
                "budget_chunks": budget_chunks,
                "hit_budget": list(hit_budget),
                "budget_retention_pct": round(budget_retention_pct, 1),
                "context_pythai_words": pythai_words,
                "context_char_count": char_count,
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_latency_sec": llm_latency,
                "retrieval_latency_sec": ret_latency,
                "budget_omission": is_omitted,
                "answer": answer,
                "answer_snippet": answer[:150].replace("\n", " ") + "..."
            }
            all_raw_results.append(record)
            strat_results.append(record)

            print(f"[{qid}] k={k} | RetTargets: {target_retention_pct:.1f}% | BudgetRet: {budget_retention_pct:.1f}% | Tokens: {prompt_tokens} | Omission: {is_omitted} | Lat: {llm_latency}s")

    # -------------------------------------------------------------------------
    # AGGREGATION & REPORTING
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 EMPIRICAL SUMMARY TABLE (REAL MEASURED DATA)")
    print("=" * 80)

    # Export Raw JSON
    json_path = PROJECT_ROOT / "exp3_topk_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_results, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved raw experimental data to {json_path}")

    # Build Markdown Summary Table
    # Calculate group averages per strategy
    summary_rows = []
    for strat in strategies:
        s_name = strat["name"]
        for grp_prefix in ["Group A", "Group B", "Group C", "Overall"]:
            if grp_prefix == "Overall":
                records = [r for r in all_raw_results if r["strategy"] == s_name]
            else:
                records = [r for r in all_raw_results if r["strategy"] == s_name and r["group"].startswith(grp_prefix)]
            
            avg_target_ret = sum(r["target_retention_pct"] for r in records) / len(records)
            avg_budget_ret = sum(r["budget_retention_pct"] for r in records) / len(records)
            avg_tokens = sum(r["llm_prompt_tokens"] for r in records) / len(records)
            avg_latency = sum(r["llm_latency_sec"] for r in records) / len(records)
            omission_rate = (sum(1 for r in records if r["budget_omission"]) / len(records)) * 100.0

            summary_rows.append({
                "Strategy": s_name,
                "Query Group": grp_prefix,
                "Avg Target Ret (%)": f"{avg_target_ret:.1f}%",
                "Avg Budget Ret (%)": f"{avg_budget_ret:.1f}%",
                "Avg Prompt Tokens": f"{avg_tokens:.0f}",
                "Omission Rate (%)": f"{omission_rate:.1f}%",
                "Avg LLM Latency (s)": f"{avg_latency:.2f}s"
            })

    md_table = "| Strategy | Query Group | Avg Target Ret (%) | Avg Budget Ret (%) | Avg Prompt Tokens | Omission Rate (%) | Avg Latency (s) |\n"
    md_table += "|---|---|---|---|---|---|---|\n"
    for row in summary_rows:
        md_table += f"| {row['Strategy']} | {row['Query Group']} | {row['Avg Target Ret (%)']} | {row['Avg Budget Ret (%)']} | {row['Avg Prompt Tokens']} | {row['Omission Rate (%)']} | {row['Avg LLM Latency (s)']} |\n"

    print("\n" + md_table)

    # Save summary table to file
    summary_path = PROJECT_ROOT / "exp3_topk_summary_table.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# 🧪 Experiment 3: Context Bottleneck Benchmark Results\n\n")
        f.write(md_table)
        f.write("\n\n## 📝 Detailed Query-by-Query Breakdown\n\n")
        f.write("| Query ID | Strategy | K | Target Chunks Hit | Budget Retention | Prompt Tokens | Omission | Latency |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in all_raw_results:
            f.write(f"| {r['query_id']} | {r['strategy']} | {r['k']} | {r['target_retention_pct']}% | {r['budget_retention_pct']}% | {r['llm_prompt_tokens']} | {'⚠️ Yes' if r['budget_omission'] else '✅ No'} | {r['llm_latency_sec']}s |\n")

    print(f"✅ Saved markdown summary table to {summary_path}")

    # Also export CSV
    csv_path = PROJECT_ROOT / "exp3_topk_summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        headers = ["strategy", "query_id", "group", "k", "target_retention_pct", "budget_retention_pct", "llm_prompt_tokens", "budget_omission", "llm_latency_sec"]
        f.write(",".join(headers) + "\n")
        for r in all_raw_results:
            row = [str(r[h]) for h in headers]
            f.write(",".join(row) + "\n")
    print(f"✅ Saved CSV to {csv_path}")

    return all_raw_results


if __name__ == "__main__":
    run_topk_bottleneck_experiment()
