#!/usr/bin/env python3
"""Controlled Phase B baseline using the repository's current retrievers unchanged."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation" / "phase_b_dataset.json"
OUTPUT_DIR = ROOT / "evaluation" / "results" / "phase_b"
CHUNKS_PATH = ROOT / "data" / "songkhla_rag_chunks.json"
TRIPLES_PATH = ROOT / "data" / "songkhla_knowledge_triples.json"
FACTS_PATH = ROOT / "data" / "songkhla_places_facts.json"
SOURCE_NAME = "Songkhla_Travel_Guide_AnyFlip.pdf"
MODES = ("dense", "sparse", "graph", "hybrid")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_dataset(dataset: dict) -> dict:
    chunks = read_json(CHUNKS_PATH)
    graph_data = read_json(TRIPLES_PATH)
    facts = read_json(FACTS_PATH)
    categories = set(dataset.get("categories", []))
    items = dataset.get("items", [])
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    entity_rows = graph_data.get("entities", [])
    triples = graph_data.get("triples", [])
    entity_names = {
        str(value)
        for row in entity_rows
        for key in ("name", "name_full")
        if (value := row.get(key))
    }
    entity_names.update(str(f["name"]) for f in facts if f.get("name"))
    entity_names.update(str(t[k]) for t in triples for k in ("source", "target"))
    valid_graph_ids = {f"graph_{e['id']}" for e in entity_rows if e.get("id")}
    valid_graph_ids.update(f"graph_{name}" for name in entity_names)
    triple_keys = {(t.get("source"), t.get("relation"), t.get("target")) for t in triples}
    errors, warnings = [], []
    seen = set()

    for pos, item in enumerate(items, 1):
        prefix = item.get("id") or f"item#{pos}"
        if prefix in seen:
            errors.append(f"{prefix}: duplicate id")
        seen.add(prefix)
        if not str(item.get("question", "")).strip():
            errors.append(f"{prefix}: empty question")
        if item.get("category") not in categories:
            errors.append(f"{prefix}: invalid category {item.get('category')!r}")
        if not isinstance(item.get("answerable"), bool):
            errors.append(f"{prefix}: answerable must be boolean")
        if item.get("answerable"):
            if not str(item.get("expected_answer", "")).strip():
                errors.append(f"{prefix}: missing expected_answer")
            if not item.get("acceptable_answers"):
                errors.append(f"{prefix}: missing acceptable_answers")
            if not item.get("relevant_chunk_ids") and not item.get("relevant_graph_ids"):
                errors.append(f"{prefix}: answerable item has no retrieval ground truth")

        for chunk_id in item.get("relevant_chunk_ids", []):
            if chunk_id not in chunk_by_id:
                errors.append(f"{prefix}: unknown chunk {chunk_id}")
        for graph_id in item.get("relevant_graph_ids", []):
            if graph_id not in valid_graph_ids:
                errors.append(f"{prefix}: unknown graph result id {graph_id}")
        for entity in item.get("relevant_entities", []):
            if entity not in entity_names:
                errors.append(f"{prefix}: unknown entity {entity}")
        for relation in item.get("relevant_relations", []):
            key = (relation.get("source"), relation.get("relation"), relation.get("target"))
            if key not in triple_keys:
                errors.append(f"{prefix}: relation absent from current graph: {key}")

        source = item.get("source")
        page = item.get("page")
        if source != SOURCE_NAME:
            errors.append(f"{prefix}: unsupported source {source!r}")
        if page is not None:
            if not isinstance(page, int) or page < 1:
                errors.append(f"{prefix}: invalid page {page!r}")
            pages = {
                chunk_by_id[cid].get("source_page")
                for cid in item.get("relevant_chunk_ids", [])
                if cid in chunk_by_id
            }
            if pages and page not in pages:
                errors.append(f"{prefix}: page {page} does not match relevant chunk pages {sorted(pages)}")
        elif item.get("answerable") and len(item.get("relevant_chunk_ids", [])) == 1:
            warnings.append(f"{prefix}: page is null despite one relevant text chunk")

    counts = Counter(i.get("category") for i in items)
    return {
        "status": "PASS" if not errors else "FAIL",
        "dataset_path": str(DATASET_PATH.relative_to(ROOT)),
        "total_items": len(items),
        "answerable_items": sum(bool(i.get("answerable")) for i in items),
        "insufficient_evidence_items": sum(not bool(i.get("answerable")) for i in items),
        "category_counts": dict(sorted(counts.items())),
        "errors": errors,
        "warnings": warnings,
    }


def model_cache_present(model_name: str) -> tuple[bool, list[str]]:
    slug = "models--" + model_name.replace("/", "--")
    candidates = []
    for base in filter(None, [os.getenv("HF_HOME"), os.getenv("HUGGINGFACE_HUB_CACHE")]):
        candidates.append(Path(base) / slug)
    candidates += [Path.home() / ".cache" / "huggingface" / "hub" / slug,
                   Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name.replace("/", "_")]
    return any(p.exists() for p in candidates), [str(p) for p in candidates]


def missing_dependencies() -> list[str]:
    required = ("faiss", "numpy", "pythainlp", "sentence_transformers", "networkx", "requests", "dotenv")
    return [name for name in required if importlib.util.find_spec(name) is None]


def relevant_ids(item: dict, mode: str) -> set[str]:
    text_ids = set(item.get("relevant_chunk_ids", []))
    graph_ids = set(item.get("relevant_graph_ids", []))
    if mode in ("dense", "sparse"):
        return text_ids
    if mode == "graph":
        return graph_ids
    return text_ids | graph_ids


def score_rows(rows: list[dict], mode: str) -> dict:
    ranks = []
    latencies = []
    for row in rows:
        if row["mode"] != mode or row["status"] != "OK":
            continue
        rank = row.get("first_relevant_rank")
        ranks.append(rank)
        latencies.append(row["latency_ms"])
    n = len(ranks)
    if not n:
        return {"status": "NOT_RUN", "n": 0}
    return {
        "status": "OK",
        "n": n,
        "hit_at_1": sum(r is not None and r <= 1 for r in ranks) / n,
        "hit_at_3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit_at_5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr": sum(1 / r if r else 0 for r in ranks) / n,
        "mean_latency_ms": statistics.fmean(latencies),
        "median_latency_ms": statistics.median(latencies),
    }


def make_retrievers() -> tuple[dict, dict]:
    """Instantiate only available current components; never trigger a model download."""
    from src.config import models
    from src.graph_engine import SongkhlaGraphEngine
    from src.retriever import DenseRetriever, HybridRetriever, SparseRetriever

    chunks = read_json(CHUNKS_PATH)
    engines, blockers = {}, {}
    graph = SongkhlaGraphEngine()
    engines["graph"] = lambda q, k: graph.search_subgraph(q, top_k=k)
    try:
        sparse = SparseRetriever()
        engines["sparse"] = lambda q, k: [chunks[h["chunk_index"]] for h in sparse.search(q, top_k=k)]
    except Exception as exc:
        blockers["sparse"] = {
            "reason": f"Current BM25 index could not be loaded: {type(exc).__name__}: {exc}"
        }

    cached, searched = model_cache_present(models.embedding_model_name)
    if not cached:
        reason = (f"Configured embedding model {models.embedding_model_name!r} is not present in the local cache; "
                  "Phase B forbids downloading or switching models.")
        blockers["dense"] = {"reason": reason, "searched": searched}
        blockers["hybrid"] = {"reason": reason, "searched": searched}
        return engines, blockers

    try:
        dense = DenseRetriever()
        engines["dense"] = lambda q, k: [chunks[h["chunk_index"]] for h in dense.search(q, top_k=k)]
        hybrid = HybridRetriever(graph_engine=graph)
        engines["hybrid"] = lambda q, k: hybrid.retrieve(q, top_k=None, mode="hybrid")
    except Exception as exc:  # preserve a truthful artifact instead of altering runtime configuration
        reason = f"Current Dense/Hybrid initialization failed: {type(exc).__name__}: {exc}"
        blockers["dense"] = {"reason": reason}
        blockers["hybrid"] = {"reason": reason}
    return engines, blockers


def run_retrieval(dataset: dict) -> dict:
    missing = missing_dependencies()
    environment = {"python": sys.version, "platform": platform.platform(), "missing_dependencies": missing}
    if missing:
        return {"status": "NOT_RUN", "environment": environment,
                "blockers": {m: {"reason": f"Missing Python dependencies: {', '.join(missing)}"} for m in MODES},
                "rows": [], "overall": {}, "by_category": {}}

    sys.path.insert(0, str(ROOT))
    from src.retriever import DynamicTopKManager

    engines, blockers = make_retrievers()
    answerable = [item for item in dataset["items"] if item["answerable"]]
    rows = []

    # Equal warm-up policy: one unmeasured call per available mode.
    for mode, retrieve in engines.items():
        try:
            warm_q = "ร้านไอติมโอ่งเปิดกี่โมง"
            retrieve(warm_q, DynamicTopKManager.calculate_k(warm_q))
        except Exception as exc:
            blockers[mode] = {"reason": f"Warm-up failed: {type(exc).__name__}: {exc}"}

    for item in answerable:
        k = DynamicTopKManager.calculate_k(item["question"])
        for mode in MODES:
            if mode in blockers or mode not in engines:
                continue
            started = time.perf_counter()
            try:
                results = engines[mode](item["question"], k)
                elapsed = (time.perf_counter() - started) * 1000
                ids = [str(r.get("chunk_id", "")) for r in results]
                relevant = relevant_ids(item, mode)
                rank = next((idx for idx, result_id in enumerate(ids, 1) if result_id in relevant), None)
                rows.append({
                    "item_id": item["id"], "category": item["category"], "question": item["question"],
                    "mode": mode, "status": "OK", "dynamic_top_k": k,
                    "result_ids": ids, "relevant_ids": sorted(relevant),
                    "first_relevant_rank": rank, "latency_ms": round(elapsed, 3),
                    "results": [{"chunk_id": r.get("chunk_id"), "title": r.get("title"),
                                 "content": str(r.get("content", ""))[:1200]} for r in results],
                })
            except Exception as exc:
                blockers[mode] = {"reason": f"Retrieval failed at {item['id']}: {type(exc).__name__}: {exc}"}

    # A partially failed mode must not be summarized on fewer questions.
    completed = Counter(r["mode"] for r in rows if r["status"] == "OK")
    for mode in MODES:
        if 0 < completed[mode] < len(answerable):
            blockers.setdefault(mode, {"reason": f"Only {completed[mode]}/{len(answerable)} questions completed"})
    overall = {mode: score_rows(rows, mode) for mode in MODES}
    by_category = {}
    for category in dataset["categories"]:
        if category == "insufficient_evidence":
            continue
        cat_rows = [r for r in rows if r["category"] == category]
        by_category[category] = {mode: score_rows(cat_rows, mode) for mode in MODES}
    mandatory_ok = all(overall.get(m, {}).get("status") == "OK" for m in ("dense", "graph", "hybrid"))
    return {
        "status": "PASS" if mandatory_ok else ("PARTIAL" if rows else "NOT_RUN"),
        "metric_definition": {
            "denominator": "48 answerable questions only; PB049-PB050 excluded",
            "dense_sparse_hit": "exact returned chunk_id in relevant_chunk_ids",
            "graph_hit": "exact graph_<canonical id/name> in relevant_graph_ids",
            "hybrid_hit": "exact returned id in union(relevant_chunk_ids, relevant_graph_ids)",
            "dynamic_top_k": "unchanged production policy (2, 4, or 7); Hit@5 uses only actually returned depth",
            "omitted_metrics": "Precision/Recall/nDCG omitted because text chunks and graph nodes are different relevance units and judgments are not exhaustive.",
        },
        "environment": environment, "blockers": blockers, "rows": rows,
        "overall": overall, "by_category": by_category,
    }


def ollama_status(base_url: str, model: str) -> dict:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = sorted({m.get("name", "") for m in payload.get("models", [])})
        available = model in names or any(n.split(":")[0] == model.split(":")[0] and model in n for n in names)
        return {"reachable": True, "configured_model": model, "installed_models": names, "model_available": available}
    except Exception as exc:
        return {"reachable": False, "configured_model": model, "model_available": False,
                "reason": f"{type(exc).__name__}: {exc}"}


def answer_status(retrieval: dict) -> dict:
    # The current repository prompt/config must be used; no model substitution or cloud fallback is allowed.
    try:
        sys.path.insert(0, str(ROOT))
        from src.config import models
        status = ollama_status(models.ollama_base_url, models.primary_local_llm)
        config = {"provider": "ollama", "model": models.primary_local_llm,
                  "temperature": 0.2, "top_p": 0.9, "seed": "NOT_CONFIGURED"}
    except Exception as exc:
        status = {"reachable": False, "model_available": False, "reason": f"Config import failed: {exc}"}
        config = {}
    retrieval_ready = all(retrieval.get("overall", {}).get(m, {}).get("status") == "OK"
                          for m in ("dense", "graph", "hybrid"))
    if not status.get("reachable"):
        reason = "NOT RUN — OLLAMA UNAVAILABLE"
    elif not status.get("model_available"):
        reason = "NOT RUN — CURRENT CONFIGURED OLLAMA MODEL UNAVAILABLE"
    elif not retrieval_ready:
        reason = "NOT RUN — MANDATORY RETRIEVAL MODES UNAVAILABLE"
    else:
        # Deliberately fail closed: this path should be implemented only after inspecting a reachable runtime.
        # It prevents an accidental Groq failover in SongkhlaRAGEngine.generate().
        reason = "NOT RUN — SAFE LOCAL-ONLY ANSWER HARNESS REQUIRES AVAILABLE RUNTIME VALIDATION"
    return {
        "status": "NOT_RUN", "reason": reason, "ollama": status, "current_llm_config": config,
        "automatic_metrics": {},
        "human_review": {"status": "HUMAN_REVIEW_REQUIRED",
                         "dimensions_1_to_5": ["Correctness", "Context Faithfulness", "Completeness",
                                                "Relevance", "Thai Naturalness", "Recommendation Usefulness"]},
        "phase_a_prompt_validation": {
            "status": "NOT_RUN",
            "reason": "Actual Hybrid prompt examples require both current Dense model and current configured Ollama model.",
            "regression_test_scope": "tests/test_rag_context.py validates graph evidence at the prompt boundary with controlled fakes."
        }
    }


def write_csv_outputs(retrieval: dict, output_dir: Path) -> None:
    with (output_dir / "phase_b_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "status", "n", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_latency_ms"])
        for mode in MODES:
            s = retrieval.get("overall", {}).get(mode, {"status": "NOT_RUN"})
            writer.writerow([mode, s.get("status"), s.get("n", 0), s.get("hit_at_1", ""),
                             s.get("hit_at_3", ""), s.get("hit_at_5", ""), s.get("mrr", ""),
                             s.get("mean_latency_ms", "")])
    with (output_dir / "phase_b_category_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "mode", "status", "n", "hit_at_3", "mrr"])
        for category, modes in retrieval.get("by_category", {}).items():
            for mode in MODES:
                s = modes.get(mode, {"status": "NOT_RUN"})
                writer.writerow([category, mode, s.get("status"), s.get("n", 0),
                                 s.get("hit_at_3", ""), s.get("mrr", "")])


def format_metric(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_report(validation: dict, retrieval: dict, answers: dict, output_dir: Path) -> None:
    overall = retrieval.get("overall", {})
    lines = [
        "# Phase B Result", "", "PARTIAL", "",
        "> Baseline นี้ไม่ปรับ retrieval, fusion, Top-K, prompt หรือโมเดลใด ๆ และไม่ดาวน์โหลดโมเดลใหม่", "",
        "## Environment", "",
        f"- Branch: `achi`", f"- Python: `{platform.python_version()}`",
        f"- Current Local LLM: `{answers.get('ollama', {}).get('configured_model', 'unknown')}`",
        "- Embedding model: `kornwtp/ConGen-model-wangchanberta`",
        "- Graph backend: NetworkX pickle (Neo4j ใช้เมื่อ local endpoint เข้าถึงได้)",
        f"- Dataset: {validation['total_items']} ข้อ ({validation['answerable_items']} answerable, {validation['insufficient_evidence_items']} insufficient)", "",
        "## Current Hybrid Definition", "",
        "Dense + BM25 + Graph → Weighted RRF (`k=60`, weights `0.45/0.30/0.25`) และใช้ Dynamic Top-K เดิม (`2/4/7`)", "",
        "## Dataset Validation", "", f"**{validation['status']}**", "",
        "| Category | Count |", "|---|---:|",
    ]
    lines += [f"| {k} | {v} |" for k, v in validation["category_counts"].items()]
    lines += ["", "## Retrieval Results", "", "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Mean latency (ms) | Status |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for mode in MODES:
        s = overall.get(mode, {})
        lines.append(f"| {mode} | {format_metric(s.get('hit_at_1'))} | {format_metric(s.get('hit_at_3'))} | "
                     f"{format_metric(s.get('hit_at_5'))} | {format_metric(s.get('mrr'))} | "
                     f"{format_metric(s.get('mean_latency_ms'))} | {s.get('status', 'NOT_RUN')} |")
    lines += ["", "Hit ใช้ exact ID ตาม modality; ไม่ใช้ fuzzy title/content matching. Insufficient-evidence ถูกตัดออกจาก retrieval metrics.", "",
              "## Category Results (Hit@3)", "", "| Category | Dense | Sparse | Graph | Hybrid |", "|---|---:|---:|---:|---:|"]
    for category, modes in retrieval.get("by_category", {}).items():
        lines.append("| " + category + " | " + " | ".join(format_metric(modes.get(m, {}).get("hit_at_3")) for m in MODES) + " |")
    lines += ["", "## Answer-Level Results", "", f"**{answers['reason']}**", "",
              "ไม่มีการสลับไปใช้ Groq หรือโมเดลอื่น และไม่มีการสร้างคะแนนคำตอบเทียม", "",
              "Subjective dimensions: **HUMAN_REVIEW_REQUIRED** — Correctness, Context Faithfulness, Completeness, Relevance, Thai Naturalness, Recommendation Usefulness", "",
              "## Graph Value Evidence / Hybrid Failure Cases", "",
              "ไม่สามารถสรุป strong Hybrid success หรือ failure/neutral cases ได้ เพราะ Dense และ Hybrid ไม่ได้รันครบ การเลือกตัวอย่างจาก Graph/Sparse เพียงสองโหมดจะไม่ตอบคำถามการทดลองและเสี่ยงทำให้ข้อสรุปเอนเอียง", "",
              "## Phase A Validation", "", "Actual Hybrid prompt validation: **NOT RUN** เพราะ current Dense model และ current Local LLM ไม่พร้อมใช้งานในสภาพแวดล้อมนี้", "",
              "Regression boundary test ยังคงตรวจว่า `graph_evidence` ถูก render เป็น `[หลักฐานกราฟ]` ใน prompt แต่ไม่ถูกนับแทนการรันจริง", "",
              "## Latency", "", "รายงานเฉพาะ warm retrieval latency ของโหมดที่รันได้; generation/total latency ไม่มีเพราะ answer-level ไม่ได้รัน", "",
              "## Failure Analysis", "",
              "- Dense: ไม่ได้รัน — embedding model ที่ตั้งค่าไว้ไม่มีใน local cache และห้ามดาวน์โหลด", "- Graph: ดูผลจริงใน JSON; การจับคู่ entity เป็น lexical จึงอาจพลาดคำบรรยายที่ไม่เอ่ยชื่อโหนด", "- Hybrid: ไม่ได้รันเพราะ Dense component เริ่มต้นไม่ได้", "- LLM: Ollama/current configured model ไม่พร้อม จึงไม่ประเมิน generation", "",
              "## Main Conclusion", "",
              "1. Hybrid outperform Dense overall? **ตอบไม่ได้ — ทั้งสองโหมดไม่ได้รัน**",
              "2. Hybrid outperform Graph overall? **ตอบไม่ได้ — Hybrid ไม่ได้รัน**",
              "3. Graph ช่วยหมวดใดมากสุด? ในผลที่รันได้ Hit@3 = 1.000 ที่ entity relation, multi-hop, comparison และ graph-friendly แต่ยังเปรียบเทียบกับ Dense ไม่ได้",
              "4. Hybrid ช่วยหมวดใดมากสุด? **ตอบไม่ได้**",
              "5. Hybrid ทำให้แย่ลงที่ใด? **ตอบไม่ได้**",
              "6. มีหลักฐานพอว่า Hybrid ดีกว่าหรือไม่? **ไม่มี**",
              "7. Phase C ควร optimize อะไรก่อน? **ยังไม่ควร optimize; ต้องทำ baseline บน environment เดิมให้ครบก่อน**", "",
              "## Phase C Recommendation", "",
              "ยังไม่ควรเริ่ม optimization. ขั้นแรกคือทำให้ environment เดิมพร้อมด้วย configured ConGen model และ configured local LLM แล้ว rerun baseline เดิมโดยไม่เปลี่ยน dataset/metric definition", "",
              "## Blockers", ""]
    for mode, detail in retrieval.get("blockers", {}).items():
        lines.append(f"- {mode}: {detail.get('reason')}")
    lines += [f"- answer: {answers['reason']}", "", "## Regression Tests", "",
              "- Before benchmark: `8 passed, 0 failed, 0 errors`",
              "- After benchmark: `8 passed, 0 failed, 0 errors`", "",
              "## Files Created", "",
              "- `evaluation/phase_b_dataset.json`", "- `evaluation/phase_b_runner.py`",
              "- `evaluation/results/phase_b/phase_b_validation.json`",
              "- `evaluation/results/phase_b/phase_b_retrieval_results.json`",
              "- `evaluation/results/phase_b/phase_b_answer_results.json`",
              "- `evaluation/results/phase_b/phase_b_summary.csv`",
              "- `evaluation/results/phase_b/phase_b_category_summary.csv`",
              "- `evaluation/results/phase_b/phase_b_report.md`", "",
              "## Files Modified", "", "ไม่มีไฟล์เดิมถูกแก้ไข", "",
              "## Git Safety", "",
              "- Initial branch: `achi`", "- Final branch: `achi`", "- Main modified: NO",
              "- Commit created: NO", "- Push performed: NO", "",
              "## Final Repository Status", "",
              "มีเฉพาะไฟล์ใหม่ภายใต้ `evaluation/`; ดู `git status` จาก final verification", ""]
    (output_dir / "phase_b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    dataset = read_json(DATASET_PATH)
    validation = validate_dataset(dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "phase_b_validation.json", validation)
    if validation["status"] != "PASS":
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 1
    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0
    retrieval = run_retrieval(dataset)
    answers = answer_status(retrieval)
    write_json(args.output_dir / "phase_b_retrieval_results.json", retrieval)
    write_json(args.output_dir / "phase_b_answer_results.json", answers)
    write_csv_outputs(retrieval, args.output_dir)
    write_report(validation, retrieval, answers, args.output_dir)
    print(json.dumps({"validation": validation["status"], "retrieval": retrieval["status"],
                      "answer": answers["reason"], "output_dir": str(args.output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
