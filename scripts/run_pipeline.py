# -*- coding: utf-8 -*-
"""
run_pipeline.py
Master Pipeline Orchestrator for Songkhla Old Town AI Assistant.
Course: 241-351 AI for Social Media (PSU Final Project)

Usage:
  python finalproject/scripts/run_pipeline.py --all           # Run full pipeline (01 to 07)
  python finalproject/scripts/run_pipeline.py --step <1-7>     # Run a specific step
  python finalproject/scripts/run_pipeline.py --verify        # Run step 07 verification
  python finalproject/scripts/run_pipeline.py --bench         # Run step 06 benchmark
  python finalproject/scripts/run_pipeline.py --index         # Rebuild data & indices (01 to 05)
"""

import sys
import os
import time
import argparse
import subprocess
from pathlib import Path

# Script registry with clear descriptions and outputs
STEPS = {
    1: {
        "file": "pipeline/01_enrich_places.py",
        "name": "Places Knowledge Base Enrichment",
        "desc": "Compiles verified Google Knowledge Panel facts, opening hours, phones, coordinates",
        "outputs": ["data/songkhla_places_facts.json", "data/Songkhla_Old_Town_Factsheet.md"]
    },
    2: {
        "file": "pipeline/02_extract_and_chunk.py",
        "name": "Data Cleaning & Semantic Chunking",
        "desc": "Cleans AnyFlip text, strips Chinese noise, performs semantic chunking with facts",
        "outputs": ["data/songkhla_rag_chunks.json", "data/songkhla_rag_chunks.md"]
    },
    3: {
        "file": "pipeline/03_extract_ontology_triples.py",
        "name": "Ontology & Knowledge Triples Construction",
        "desc": "Builds domain ontology schema and extracts multi-relational knowledge triples",
        "outputs": ["data/songkhla_ontology_schema.json", "data/songkhla_knowledge_triples.json"]
    },
    4: {
        "file": "pipeline/04_build_graph.py",
        "name": "Knowledge Graph Modeling & Visualization",
        "desc": "Builds NetworkX/Neo4j graph, calculates PageRank/centrality, exports PyVis HTML & PNG",
        "outputs": ["data/songkhla_graph.pkl", "data/songkhla_knowledge_graph.html", "data/songkhla_graph_overview.png"]
    },
    5: {
        "file": "pipeline/05_build_indices.py",
        "name": "Hybrid Retrieval Indexing (Dense FAISS + Sparse BM25)",
        "desc": "Encodes WangchanBERTa embeddings into FAISS and tokenizes PyThaiNLP into BM25",
        "outputs": ["data/songkhla_faiss.index", "data/songkhla_bm25.pkl", "data/songkhla_chunks_metadata.json"]
    },
    6: {
        "file": "experiments/06_benchmark_evaluation.py",
        "name": "Empirical Benchmark & Evaluation Suite",
        "desc": "Evaluates 15 in-domain queries across Dense, Sparse, Graph, Hybrid RAG + Latency & MRR",
        "outputs": ["data/benchmark_results.json", "data/benchmark_summary_table.md"]
    },
    7: {
        "file": "tools/07_test_end_to_end.py",
        "name": "End-to-End System & LINE Schema Validation",
        "desc": "Verifies intent classification, Local Ollama generation, multi-message carousels, LINE API",
        "outputs": ["Live LINE Schema Verification & System Status"]
    }
}


def print_banner():
    banner = """
================================================================================
🏛️  SONGKHLA OLD TOWN ASSISTANT - AI PIPELINE RUNNER
📍  Project: GraphRAG + Dual LLM for Songkhla Cultural Tourism
🎓  Course: 241-351 AI for Social Media | Prince of Songkla University
================================================================================
"""
    print(banner)


def run_step(step_num: int, python_exe: str, scripts_dir: Path) -> bool:
    info = STEPS.get(step_num)
    if not info:
        print(f"❌ Invalid step number: {step_num}")
        return False

    script_path = scripts_dir / info["file"]
    if not script_path.exists():
        print(f"❌ Script file not found: {script_path}")
        return False

    print("\n" + "-" * 80)
    print(f"▶️  STEP {step_num:02d}: {info['name']}")
    print(f"    ℹ️  {info['desc']}")
    print(f"    📄 Script: {info['file']}")
    print("-" * 80)

    t0 = time.time()
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"

    proc = subprocess.run([python_exe, str(script_path)], env=env)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\n✅ [STEP {step_num:02d} COMPLETED] Duration: {elapsed:.2f}s")
        print("   📦 Expected Outputs:")
        for out in info["outputs"]:
            print(f"      - {out}")
        return True
    else:
        print(f"\n❌ [STEP {step_num:02d} FAILED] Exit Code: {proc.returncode} (Duration: {elapsed:.2f}s)")
        return False


def main():
    print_banner()
    parser = argparse.ArgumentParser(description="Master Pipeline Runner for Songkhla Old Town AI Assistant")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline from Step 01 to Step 07")
    parser.add_argument("--step", type=int, choices=range(1, 8), help="Run a single specific step (1 to 7)")
    parser.add_argument("--verify", action="store_true", help="Run Step 07 (End-to-End System & LINE API Validation)")
    parser.add_argument("--bench", action="store_true", help="Run Step 06 (Empirical Benchmark Evaluation)")
    parser.add_argument("--index", action="store_true", help="Rebuild all knowledge and search indices (Steps 01 to 05)")
    args = parser.parse_args()

    # Determine Python executable
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent.parent
    venv_python = project_root / "venv" / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    # Select steps to run
    steps_to_run = []
    if args.all:
        steps_to_run = list(range(1, 8))
    elif args.verify:
        steps_to_run = [7]
    elif args.bench:
        steps_to_run = [6]
    elif args.index:
        steps_to_run = [1, 2, 3, 4, 5]
    elif args.step:
        steps_to_run = [args.step]
    else:
        # Default interactive menu or show help
        print("Please specify an option to execute the pipeline:\n")
        print("  --all       : Run all 7 stages from data ingestion to verification")
        print("  --verify    : Run end-to-end integration and LINE schema test")
        print("  --bench     : Run retrieval and LLM benchmark suite")
        print("  --index     : Rebuild all data, graphs, and FAISS/BM25 indices (1-5)")
        print("  --step <N>  : Run a single specific stage (1 to 7)\n")
        print("Example: python finalproject/scripts/run_pipeline.py --verify")
        return

    print(f"Target Python: {python_exe}")
    print(f"Stages to execute: {steps_to_run}\n")

    overall_t0 = time.time()
    summary = []

    for step in steps_to_run:
        ok = run_step(step, python_exe, scripts_dir)
        summary.append((step, STEPS[step]["name"], "PASS" if ok else "FAIL"))
        if not ok and len(steps_to_run) > 1:
            print(f"\n⛔ Pipeline halted due to error in Step {step:02d}.")
            break

    total_time = time.time() - overall_t0

    print("\n" + "=" * 80)
    print("🏁 PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for step, name, status in summary:
        mark = "✅ PASS" if status == "PASS" else "❌ FAIL"
        print(f"  Step {step:02d}: {name:<45} | {mark}")
    print("-" * 80)
    print(f"⏱️  Total Execution Time: {total_time:.2f} seconds")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
