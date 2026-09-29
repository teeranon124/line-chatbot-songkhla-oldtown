#!/usr/bin/env python3
"""Run the frozen Phase B.1 controlled baseline without changing production code."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "phase_b_dataset.json"
OUTPUT = ROOT / "evaluation" / "results" / "phase_b_complete"
EXPECTED_HASH = "4e1101d07e5f3d1ee49c22ba87a7d984179b3af8dfb37936242a944330526e76"
MODES = ("dense", "sparse", "graph", "hybrid")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text: str) -> str:
    return "".join(str(text).lower().split()).replace("-", "").replace(":", "")


def relevant_ids(item: dict, mode: str) -> set[str]:
    text_ids = set(item.get("relevant_chunk_ids", []))
    graph_ids = set(item.get("relevant_graph_ids", []))
    if mode in ("dense", "sparse"):
        return text_ids
    if mode == "graph":
        return graph_ids
    return text_ids | graph_ids


def rank_of(ids: list[str], relevant: set[str]):
    return next((i for i, value in enumerate(ids, 1) if value in relevant), None)


def summarize(rows: list[dict], mode: str) -> dict:
    selected = [r for r in rows if r["mode"] == mode and r["answerable"]]
    ranks = [r["first_relevant_rank"] for r in selected]
    latency = [r["retrieval_latency_ms"] for r in selected]
    n = len(selected)
    return {
        "n": n,
        "hit_at_1": sum(r is not None and r <= 1 for r in ranks) / n,
        "hit_at_3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r if r else 0 for r in ranks) / n,
        "mean_latency_ms": statistics.fmean(latency),
        "median_latency_ms": statistics.median(latency),
    }


class StaticRetriever:
    """In-memory adapter allowing production generate() to use pre-retrieved evidence."""

    def __init__(self):
        self.chunks = []

    def retrieve(self, query, top_k=None, mode="hybrid"):
        return self.chunks


def answer_summary(rows: list[dict], mode: str) -> dict:
    answerable = [r for r in rows if r["mode"] == mode and r["answerable"]]
    insufficient = [r for r in rows if r["mode"] == mode and not r["answerable"]]
    generation = [r["generation_latency_ms"] for r in rows if r["mode"] == mode]
    total = [r["total_latency_ms"] for r in rows if r["mode"] == mode]
    return {
        "answerable_n": len(answerable),
        "automatic_acceptable_phrase_hit": sum(r["automatic_acceptable_phrase_hit"] for r in answerable) / len(answerable),
        "automatic_evidence_supported_phrase_hit": sum(r["automatic_evidence_supported_phrase_hit"] for r in answerable) / len(answerable),
        "insufficient_n": len(insufficient),
        "refusal_rate": sum(r["insufficient_refusal"] for r in insufficient) / len(insufficient),
        "mean_generation_latency_ms": statistics.fmean(generation),
        "median_generation_latency_ms": statistics.median(generation),
        "mean_total_latency_ms": statistics.fmean(total),
        "median_total_latency_ms": statistics.median(total),
    }


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from evaluation.phase_b_runner import validate_dataset
    from src.config import models, paths, retrieval
    from src.rag_engine import SYSTEM_PROMPT, SongkhlaRAGEngine
    from src.retriever import HybridRetriever

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset_hash_before = sha256(DATASET)
    if dataset_hash_before != EXPECTED_HASH:
        raise RuntimeError(f"Frozen dataset hash mismatch: {dataset_hash_before}")
    dataset = load_json(DATASET)
    validation = validate_dataset(dataset)
    if validation["status"] != "PASS" or validation["total_items"] != 50:
        raise RuntimeError(f"Dataset validation failed: {validation}")

    print("[1/5] Initializing production retriever", flush=True)
    retriever = HybridRetriever()
    chunks_by_item_mode = {}
    retrieval_rows = []
    warm_query = "ร้านไอติมโอ่งเปิดกี่โมง"
    for mode in MODES:
        retriever.retrieve(warm_query, mode=mode)

    print("[2/5] Running 50 questions x 4 retrieval modes", flush=True)
    for item_number, item in enumerate(dataset["items"], 1):
        for mode in MODES:
            started = time.perf_counter()
            chunks = retriever.retrieve(item["question"], mode=mode)
            elapsed = (time.perf_counter() - started) * 1000
            chunks_by_item_mode[(item["id"], mode)] = chunks
            ids = [str(c.get("chunk_id", "")) for c in chunks]
            relevant = relevant_ids(item, mode)
            retrieval_rows.append({
                "item_id": item["id"], "category": item["category"], "question": item["question"],
                "answerable": item["answerable"], "mode": mode,
                "dynamic_result_count": len(chunks), "result_ids": ids,
                "relevant_ids": sorted(relevant), "first_relevant_rank": rank_of(ids, relevant),
                "retrieval_latency_ms": round(elapsed, 3),
                "results": [{
                    "chunk_id": c.get("chunk_id"), "title": c.get("title"), "category": c.get("category"),
                    "content": str(c.get("content", "")),
                    "graph_evidence": c.get("graph_evidence", []),
                } for c in chunks],
            })
        if item_number % 10 == 0:
            print(f"  retrieval {item_number}/50", flush=True)

    overall = {mode: summarize(retrieval_rows, mode) for mode in MODES}
    by_category = {}
    for category in dataset["categories"]:
        if category == "insufficient_evidence":
            continue
        cat_rows = [r for r in retrieval_rows if r["category"] == category]
        by_category[category] = {mode: summarize(cat_rows, mode) for mode in MODES}

    retrieval_output = {
        "status": "PASS", "dataset_sha256": dataset_hash_before,
        "timing_methodology": "One unmeasured warm-up per mode; model/index initialization excluded; perf_counter around retrieve() only.",
        "relevance_definition": {
            "dense": "exact returned chunk_id in frozen relevant_chunk_ids",
            "sparse": "exact returned chunk_id in frozen relevant_chunk_ids",
            "graph": "exact returned graph id in frozen relevant_graph_ids",
            "hybrid": "exact returned id in union of frozen relevant_chunk_ids and relevant_graph_ids",
            "insufficient_evidence": "excluded from Hit@K and MRR",
            "omitted_metrics": "Precision/Recall/nDCG omitted because graph nodes and text chunks are different, non-exhaustively judged relevance units.",
        },
        "configuration": {
            "embedding": models.embedding_model_name,
            "faiss_path": str(paths.faiss_index_path), "bm25_path": str(paths.bm25_index_path),
            "graph_path": str(paths.graph_pkl_path),
            "weights": {"dense": retrieval.dense_weight, "sparse": retrieval.sparse_weight, "graph": retrieval.graph_weight},
            "rrf_k": retrieval.rrf_k,
            "dynamic_top_k": {"min": retrieval.min_top_k, "default": retrieval.default_top_k, "max": retrieval.max_top_k},
        },
        "overall": overall, "by_category": by_category, "rows": retrieval_rows,
    }
    dump_json(OUTPUT / "phase_b_complete_retrieval_results.json", retrieval_output)

    print("[3/5] Warming current Ollama model", flush=True)
    engine = SongkhlaRAGEngine.__new__(SongkhlaRAGEngine)
    engine.retriever = StaticRetriever()
    engine.groq_api_key = ""  # explicitly prevent cloud failover
    engine.groq_model = models.groq_model
    engine.ollama_base_url = models.ollama_base_url
    engine.local_model = models.primary_local_llm
    engine.sessions = {}
    warm_started = time.perf_counter()
    warm_answer = engine.call_ollama([{"role": "user", "content": "ตอบว่า พร้อม"}], engine.local_model)
    warm_ms = (time.perf_counter() - warm_started) * 1000
    if warm_answer.startswith(("Ollama Error", "Local LLM Error")):
        raise RuntimeError(warm_answer)

    original_call = engine.call_ollama
    observation = {}

    def observed_call(messages, model_name):
        observation["messages"] = messages
        started = time.perf_counter()
        answer = original_call(messages, model_name)
        observation["generation_latency_ms"] = (time.perf_counter() - started) * 1000
        return answer

    engine.call_ollama = observed_call
    answer_rows = []
    prompt_candidates = []
    refusal_markers = ("ไม่มีข้อมูล", "ไม่พบข้อมูล", "ไม่ได้ระบุ", "ข้อมูลไม่เพียงพอ", "ไม่สามารถยืนยัน")

    print("[4/5] Running 50 questions x 4 answer modes", flush=True)
    retrieval_lookup = {(r["item_id"], r["mode"]): r for r in retrieval_rows}
    for item_number, item in enumerate(dataset["items"], 1):
        for mode in MODES:
            chunks = chunks_by_item_mode[(item["id"], mode)]
            engine.retriever.chunks = chunks
            observation.clear()
            total_started = time.perf_counter()
            result = engine.generate(item["question"], mode=mode, target_llm="ollama", user_id="")
            answer_total_ms = (time.perf_counter() - total_started) * 1000
            generation_ms = observation.get("generation_latency_ms", answer_total_ms)
            context = SongkhlaRAGEngine.build_context(chunks)
            answer = result["answer"]
            acceptable = item.get("acceptable_answers", [])
            matched = [term for term in acceptable if norm(term) and norm(term) in norm(answer)]
            grounded_matched = [term for term in matched if norm(term) in norm(context)]
            refusal = any(marker in answer for marker in refusal_markers)
            retrieval_ms = retrieval_lookup[(item["id"], mode)]["retrieval_latency_ms"]
            row = {
                "item_id": item["id"], "category": item["category"], "question": item["question"],
                "answerable": item["answerable"], "mode": mode, "answer": answer,
                "expected_answer": item["expected_answer"], "acceptable_answers": acceptable,
                "matched_acceptable_phrases": matched, "matched_grounded_phrases": grounded_matched,
                "automatic_acceptable_phrase_hit": bool(matched),
                "automatic_evidence_supported_phrase_hit": bool(grounded_matched),
                "insufficient_refusal": refusal if not item["answerable"] else None,
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": round(generation_ms, 3),
                "answer_pipeline_latency_ms": round(answer_total_ms, 3),
                "total_latency_ms": round(retrieval_ms + answer_total_ms, 3),
                "sources": result.get("sources", []),
            }
            answer_rows.append(row)
            if mode == "hybrid" and item["category"] in ("graph_friendly", "multi_hop_relation"):
                graph_evidence = [g for c in chunks for g in c.get("graph_evidence", [])]
                if graph_evidence:
                    user_prompt = observation.get("messages", [{}])[-1].get("content", "")
                    evidence_texts = [
                        str(g.get("content", ""))
                        for g in graph_evidence
                        if isinstance(g, dict) and str(g.get("content", "")).strip()
                    ]
                    prompt_candidates.append({
                        "item_id": item["id"], "question": item["question"],
                        "graph_evidence": graph_evidence,
                        "hybrid_result_ids": [c.get("chunk_id") for c in chunks],
                        "structured_context": context,
                        "final_user_prompt": user_prompt,
                        "evidence_present_in_final_prompt": bool(evidence_texts) and all(
                            evidence_text in user_prompt for evidence_text in evidence_texts
                        ),
                        "answer": answer,
                        "answer_grounded_in_evidence": "YES" if grounded_matched else "UNCERTAIN",
                    })
        if item_number % 5 == 0:
            print(f"  answers {item_number}/50", flush=True)

    answer_overall = {mode: answer_summary(answer_rows, mode) for mode in MODES}
    answer_output = {
        "status": "PASS", "dataset_sha256": dataset_hash_before,
        "model": models.primary_local_llm,
        "settings": {"temperature": 0.2, "top_p": 0.9, "seed": "NOT_CONFIGURED"},
        "warmup": {"response": warm_answer, "latency_ms": round(warm_ms, 3), "excluded_from_timing": True},
        "automatic_metric_limitations": (
            "Acceptable-phrase hit checks whether at least one frozen acceptable string appears; it is not semantic correctness. "
            "Evidence-supported phrase hit additionally requires that string in rendered context; it is not full groundedness."
        ),
        "subjective_quality": "HUMAN_REVIEW_REQUIRED",
        "overall": answer_overall, "rows": answer_rows,
    }
    dump_json(OUTPUT / "phase_b_complete_answer_results.json", answer_output)

    prompt_validation = {
        "status": "PASS" if len(prompt_candidates) >= 3 and all(x["evidence_present_in_final_prompt"] for x in prompt_candidates[:3]) else "FAIL",
        "selection_rule": "First three frozen graph-friendly/multi-hop Hybrid questions with structured graph evidence.",
        "examples": prompt_candidates[:3],
    }
    dump_json(OUTPUT / "phase_b_complete_prompt_validation.json", prompt_validation)

    # Measured graph value comparisons, without subjective cherry-picking.
    answer_lookup = {(r["item_id"], r["mode"]): r for r in answer_rows}
    by_item_retrieval = defaultdict(dict)
    for row in retrieval_rows:
        by_item_retrieval[row["item_id"]][row["mode"]] = row
    helps, hurts, neutral = [], [], []
    for item in dataset["items"]:
        if not item["answerable"]:
            continue
        rows = by_item_retrieval[item["id"]]
        dr, gr, hr = rows["dense"]["first_relevant_rank"], rows["graph"]["first_relevant_rank"], rows["hybrid"]["first_relevant_rank"]
        da = answer_lookup[(item["id"], "dense")]["automatic_acceptable_phrase_hit"]
        ha = answer_lookup[(item["id"], "hybrid")]["automatic_acceptable_phrase_hit"]
        record = {"item_id": item["id"], "question": item["question"], "dense_rank": dr, "graph_rank": gr,
                  "hybrid_rank": hr, "dense_phrase_hit": da, "hybrid_phrase_hit": ha}
        if gr is not None and hr is not None and (dr is None or hr < dr or (not da and ha)):
            helps.append(record)
        elif dr is not None and (hr is None or hr > dr or (da and not ha)):
            hurts.append(record)
        elif gr is not None and hr == dr and ha == da:
            neutral.append(record)

    failure_counts = {mode: Counter() for mode in MODES}
    failure_examples = {mode: defaultdict(list) for mode in MODES}
    for row in retrieval_rows:
        if not row["answerable"] or row["first_relevant_rank"] is not None:
            continue
        mode, category = row["mode"], row["category"]
        if mode == "dense":
            label = "semantic miss" if category in ("lexical_mismatch", "dense_friendly", "natural_thai", "recommendation") else "ranking failure"
        elif mode == "sparse":
            label = "vocabulary mismatch" if category in ("lexical_mismatch", "natural_thai", "recommendation") else "ranking failure"
        elif mode == "graph":
            label = "entity matching issue" if not row["result_ids"] else "graph traversal/ontology limitation"
        else:
            label = "fusion ranking or Dynamic Top-K limitation"
        failure_counts[mode][label] += 1
        if len(failure_examples[mode][label]) < 5:
            failure_examples[mode][label].append(row["item_id"])
    llm_failures = Counter()
    llm_examples = defaultdict(list)
    for row in answer_rows:
        if row["answerable"] and not row["automatic_acceptable_phrase_hit"]:
            label = "expected phrase absent (human review required)"
        elif not row["answerable"] and not row["insufficient_refusal"]:
            label = "refusal error"
        else:
            continue
        llm_failures[label] += 1
        if len(llm_examples[label]) < 8:
            llm_examples[label].append(f"{row['item_id']}:{row['mode']}")
    failure_output = {
        "classification_rule": "Evidence-based labels derived only from observed exact-ID misses and frozen category; semantic causality remains human-reviewable.",
        "retrieval": {mode: {k: {"count": v, "examples": failure_examples[mode][k]} for k, v in failure_counts[mode].items()} for mode in MODES},
        "llm": {k: {"count": v, "examples": llm_examples[k]} for k, v in llm_failures.items()},
        "graph_helps": helps, "graph_hurts": hurts, "graph_neutral": neutral,
    }
    dump_json(OUTPUT / "phase_b_complete_failure_analysis.json", failure_output)

    with (OUTPUT / "phase_b_complete_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "n", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_latency_ms", "median_latency_ms"])
        for mode in MODES:
            s = overall[mode]; w.writerow([mode, s["n"], s["hit_at_1"], s["hit_at_3"], s["hit_at_5"], s["mrr"], s["mean_latency_ms"], s["median_latency_ms"]])
    with (OUTPUT / "phase_b_complete_category_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["category", "mode", "n", "hit_at_1", "hit_at_3", "hit_at_5", "mrr"])
        for category, modes in by_category.items():
            for mode in MODES:
                s = modes[mode]; w.writerow([category, mode, s["n"], s["hit_at_1"], s["hit_at_3"], s["hit_at_5"], s["mrr"]])

    winners = {metric: max(MODES, key=lambda m: overall[m][metric]) for metric in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")}
    report = {
        "dataset_hash_before": dataset_hash_before, "dataset_hash_after": sha256(DATASET),
        "dataset_unchanged": dataset_hash_before == sha256(DATASET),
        "validation": validation, "retrieval_winners": winners,
        "prompt_validation": prompt_validation["status"], "graph_help_count": len(helps),
        "graph_hurt_count": len(hurts), "graph_neutral_count": len(neutral),
    }
    dump_json(OUTPUT / "phase_b_complete_run_manifest.json", report)
    print("[5/5] Complete", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
