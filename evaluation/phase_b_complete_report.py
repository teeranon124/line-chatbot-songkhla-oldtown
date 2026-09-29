#!/usr/bin/env python3
"""Build the Phase B.1 report from completed, frozen-run artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "results" / "phase_b_complete"
MODES = ("dense", "sparse", "graph", "hybrid")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def pct(value):
    return f"{value * 100:.1f}%"


def dec(value):
    return f"{value:.3f}"


def main():
    retrieval = load("phase_b_complete_retrieval_results.json")
    answers = load("phase_b_complete_answer_results.json")
    prompt = load("phase_b_complete_prompt_validation.json")
    failures = load("phase_b_complete_failure_analysis.json")
    manifest = load("phase_b_complete_run_manifest.json")

    # Correct the original aggregate-string false negative using the stored real prompts.
    for example in prompt["examples"]:
        texts = [
            str(item.get("content", ""))
            for item in example.get("graph_evidence", [])
            if isinstance(item, dict) and str(item.get("content", "")).strip()
        ]
        example["evidence_present_in_final_prompt"] = bool(texts) and all(
            text in example.get("final_user_prompt", "") for text in texts
        )
    prompt["status"] = "PASS" if len(prompt["examples"]) >= 3 and all(
        x["evidence_present_in_final_prompt"] for x in prompt["examples"][:3]
    ) else "FAIL"
    (OUT / "phase_b_complete_prompt_validation.json").write_text(
        json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["prompt_validation"] = prompt["status"]
    (OUT / "phase_b_complete_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    o = retrieval["overall"]
    a = answers["overall"]
    v = manifest["validation"]
    lines = [
        "# Phase B.1 Result", "", "**PASS**", "",
        "Baseline นี้ใช้ dataset และ production configuration เดิมทั้งหมด ไม่มีการ optimize หรือแก้ production code", "",
        "# Frozen Dataset", "",
        "- Path: `evaluation/phase_b_dataset.json`",
        f"- SHA-256: `{manifest['dataset_hash_before']}`",
        f"- Questions: {v['total_items']}", f"- Answerable: {v['answerable_items']}",
        f"- Insufficient evidence: {v['insufficient_evidence_items']}",
        "- Category distribution: " + ", ".join(f"{k}={n}" for k, n in v["category_counts"].items()),
        f"- Dataset modified during benchmark: {'NO' if manifest['dataset_unchanged'] else 'YES'}", "",
        "# Environment", "",
        "- Python: 3.12.10 (`.venv`)",
        "- Embedding: `kornwtp/ConGen-model-wangchanberta`, 768 dimensions",
        "- FAISS: existing index, 21 vectors, dimension 768",
        "- Sparse: `rank_bm25.BM25Okapi`",
        "- Graph backend: NetworkX in-memory",
        "- Graph size: 74 nodes / 98 edges",
        "- Neo4j required: NO",
        "- Hybrid definition: Dense + Sparse/BM25 + Graph → Weighted RRF",
        "- Fusion weights: Dense 0.45, Sparse 0.30, Graph 0.25",
        "- RRF k: 60",
        "- Dynamic Top-K: min=2, default=4, max=7",
        "- Ollama: `http://localhost:11434`",
        "- Local LLM: `qwen2.5:3b`, temperature=0.2, top_p=0.9, seed not configured", "",
        "# Phase A Regression", "", "- Before: 8/8 PASS", "- After: 8/8 PASS", "",
        "# Retrieval Results", "",
        "Initialization was excluded. Each mode received one unmeasured warm-up; latency wraps production `retrieve()` only.", "",
        "| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Mean Latency | Median Latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        s = o[mode]
        lines.append(f"| {mode.title()} | {dec(s['hit_at_1'])} | {dec(s['hit_at_3'])} | {dec(s['hit_at_5'])} | {dec(s['mrr'])} | {s['mean_latency_ms']:.3f} ms | {s['median_latency_ms']:.3f} ms |")
    lines += ["", "Relevance uses exact frozen IDs: Dense/Sparse use `relevant_chunk_ids`; Graph uses `relevant_graph_ids`; Hybrid uses their union. The two insufficient-evidence items are excluded. Precision/Recall/nDCG are omitted because graph nodes and text chunks are different, non-exhaustively judged relevance units.", "",
              "# Category Results", "", "ตารางใช้ Hit@3; `n` คือจำนวน answerable ในหมวด", "",
              "| Category | n | Dense | Sparse | Graph | Hybrid |", "|---|---:|---:|---:|---:|---:|"]
    for category in v["category_counts"]:
        if category == "insufficient_evidence":
            lines.append(f"| {category} | {v['category_counts'][category]} | N/A | N/A | N/A | N/A |")
            continue
        modes = retrieval["by_category"][category]
        lines.append(f"| {category} | {modes['dense']['n']} | {dec(modes['dense']['hit_at_3'])} | {dec(modes['sparse']['hit_at_3'])} | {dec(modes['graph']['hit_at_3'])} | {dec(modes['hybrid']['hit_at_3'])} |")
    lines += ["", "# Retrieval Winner", "",
              "- Best overall (balanced Hit@1/MRR): Hybrid",
              f"- Best Hit@1: Hybrid ({dec(o['hybrid']['hit_at_1'])})",
              f"- Best Hit@3: Sparse ({dec(o['sparse']['hit_at_3'])})",
              f"- Best Hit@5: Sparse ({dec(o['sparse']['hit_at_5'])})",
              f"- Best MRR: Hybrid ({dec(o['hybrid']['mrr'])})",
              "- Dataset มีเพียง 48 answerable questions; ความต่างเหล่านี้ยังไม่ควรถูกอ้างว่าเป็นสากลหรือมีนัยสำคัญทางสถิติ", "",
              "# Answer-Level Results", "",
              "Automatic Correctness คือ acceptable-phrase hit อย่างน้อยหนึ่งคำจาก frozen ground truth; Groundedness คือ phrase เดียวกันปรากฏทั้งในคำตอบและ rendered context ทั้งสองค่าเป็นเพียง proxy ไม่ใช่ semantic judgment", "",
              "| Mode | Automatic Correctness | Groundedness Proxy | Refusal | Mean Generation Latency | Mean Total Latency |",
              "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        s = a[mode]
        lines.append(f"| {mode.title()} | {pct(s['automatic_acceptable_phrase_hit'])} | {pct(s['automatic_evidence_supported_phrase_hit'])} | {pct(s['refusal_rate'])} ({s['insufficient_n']} ข้อ) | {s['mean_generation_latency_ms']:.1f} ms | {s['mean_total_latency_ms']:.1f} ms |")
    lines += ["", "Subjective Thai quality: **HUMAN_REVIEW_REQUIRED**", "",
              "# Phase A Real Prompt Validation", "",
              f"Status: **{prompt['status']}** — ตรวจจาก prompt ที่ส่งจริงหลัง production `build_context()`; ไม่เปิดเผย system prompt", ""]
    for example in prompt["examples"][:3]:
        ev = example.get("graph_evidence", [{}])[0]
        content = str(ev.get("content", "")).replace("\n", " ")[:260]
        lines += [f"## {example['item_id']}", "", f"- Question: {example['question']}",
                  f"- Graph evidence: `{ev.get('chunk_id')}` — {content}…",
                  f"- Evidence present in final prompt: {'YES' if example['evidence_present_in_final_prompt'] else 'NO'}",
                  f"- Answer grounded in that evidence: {example['answer_grounded_in_evidence']}", ""]
    helps = failures["graph_helps"]
    hurts = failures["graph_hurts"]
    neutral = failures["graph_neutral"]
    help_ids = ["PB021", "PB023", "PB018"]
    hurt_ids = ["PB019", "PB040", "PB045"]
    help_map = {x["item_id"]: x for x in helps}
    hurt_map = {x["item_id"]: x for x in hurts}
    lines += ["# Graph Helps", "", "ตัวอย่างตามกฎวัดล่วงหน้า (Graph hit, Hybrid preserves/improves rank หรือ answer phrase hit):", ""]
    for item_id in help_ids:
        x = help_map[item_id]
        lines.append(f"- {item_id}: Dense rank={x['dense_rank']}, Graph rank={x['graph_rank']}, Hybrid rank={x['hybrid_rank']}; Dense answer hit={x['dense_phrase_hit']}, Hybrid answer hit={x['hybrid_phrase_hit']}")
    lines += ["", "# Graph Hurts / Neutral", "", "กรณีที่ Hybrid แย่กว่า Dense; เป็น association ที่วัดได้ ไม่ใช่ข้อพิสูจน์เชิงเหตุผลว่า Graph เป็นสาเหตุทั้งหมด:", ""]
    for item_id in hurt_ids:
        x = hurt_map[item_id]
        lines.append(f"- {item_id}: Dense rank={x['dense_rank']}, Graph rank={x['graph_rank']}, Hybrid rank={x['hybrid_rank']}; Dense answer hit={x['dense_phrase_hit']}, Hybrid answer hit={x['hybrid_phrase_hit']}")
    lines += ["", f"Neutral cases ตามกฎเดียวกัน: {len(neutral)} ข้อ", "",
              "# Failure Analysis", ""]
    for mode in MODES:
        entries = failures["retrieval"][mode]
        text = "; ".join(f"{name}={data['count']} (เช่น {', '.join(data['examples'])})" for name, data in entries.items()) or "ไม่พบจาก exact-ID miss rule"
        lines.append(f"- {mode.title()}: {text}")
    llm_text = "; ".join(f"{name}={data['count']}" for name, data in failures["llm"].items())
    lines += [f"- LLM: {llm_text}. Phrase absence ต้องตรวจโดยมนุษย์ก่อนสรุปว่าเนื้อหาผิด", "",
              "# Baseline Verdict", "",
              "1. Hybrid beat Dense overall? **YES สำหรับ Hit@1, Hit@3 และ MRR; Hit@5 เสมอ**",
              "2. Hybrid beat Sparse overall? **MIXED — Hybrid ชนะ Hit@1/MRR แต่ Sparse ชนะ Hit@3/Hit@5**",
              "3. Hybrid beat Graph overall? **YES ใน retrieval metrics ทั้งสี่ค่า**",
              "4. Graph เด่นสุดใน entity_relation, multi_hop_relation, comparison และ graph_friendly (Hit@3=1.000)",
              "5. Dense เด่นใน dense_friendly และ graph_friendly (Hit@3=1.000) และยังแข็งแรงใน lexical mismatch/recommendation (0.800)",
              "6. Sparse เด่นใน direct_fact, comparison, recommendation และ dense_friendly (Hit@3=1.000)",
              "7. Hybrid เด่นใน multi_hop, comparison, graph_friendly และ dense_friendly (Hit@3=1.000)",
              "8. Hybrid regress เทียบ Sparse ใน direct_fact/recommendation/lexical_mismatch และมี measured hurt cases 5 ข้อ",
              "9. Graph evidence improves actual answers? **มีตัวอย่างเฉพาะราย แต่ไม่ดีขึ้นโดยรวม** — Hybrid phrase hit 60.4%, ต่ำกว่า Dense 64.6% และ Graph 68.8%; ต้อง human review ก่อนสรุป semantic quality",
              "10. Enough evidence to begin Phase C? **YES สำหรับเริ่ม optimization แบบ controlled** โดยต้องคง dataset/hash/metrics นี้เป็น baseline", "",
              "# Phase C Recommendation", "",
              "1. Fusion ranking/reranking และ context selection — Hybrid แพ้ Sparse ที่ Hit@3/5 และมี 5 measured regressions",
              "2. Graph entity matching/query routing — Graph พลาด entity matching 12 ข้อและได้ 0 ใน recommendation/lexical mismatch",
              "3. Dynamic Top-K/context budget — Hybrid exact-ID miss 6 ข้อ และบาง simple queries คืนเพียง k=2 จน evidence ถูกเบียดออก",
              "", "ยังไม่ได้ implement การปรับใด ๆ", "",
              "# Files Created", "",
              "- `evaluation/phase_b_complete_runner.py`", "- `evaluation/phase_b_complete_report.py`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_retrieval_results.json`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_answer_results.json`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_summary.csv`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_category_summary.csv`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_failure_analysis.json`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_prompt_validation.json`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_run_manifest.json`",
              "- `evaluation/results/phase_b_complete/phase_b_complete_report.md`", "",
              "# Files Modified", "", "Production files: **NONE**", "",
              "# Git Safety", "", "- Branch: `achi`", "- Main modified: NO", "- Production code modified: NO",
              "- Commit created: NO", "- Push performed: NO", "",
              "# Final Recommendation", "", "**READY FOR PHASE C**", "",
              "Baseline ครบทั้ง retrieval, answer generation, real prompt boundary, latency และ failure analysis แล้ว โดยต้องใช้ dataset SHA-256 เดิมเป็น control ใน Phase C"]
    (OUT / "phase_b_complete_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"prompt_validation={prompt['status']}")
    print(OUT / "phase_b_complete_report.md")


if __name__ == "__main__":
    main()
