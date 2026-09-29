#!/usr/bin/env python3
"""Generate presentation-ready charts from the frozen Phase B evaluation artifacts."""

from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/songkhla_chatbot_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/songkhla_chatbot_cache")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"
COMPLETE = RESULTS / "phase_b_complete"
OPTIMIZATION = RESULTS / "final_optimization"
OUTPUT = RESULTS / "presentation_charts"

RETRIEVAL_JSON = COMPLETE / "phase_b_complete_retrieval_results.json"
RETRIEVAL_CSV = COMPLETE / "phase_b_complete_summary.csv"
CATEGORY_CSV = COMPLETE / "phase_b_complete_category_summary.csv"
ANSWER_JSON = COMPLETE / "phase_b_complete_answer_results.json"
COMPLETE_REPORT = COMPLETE / "phase_b_complete_report.md"
OPTIMIZATION_CSV = OPTIMIZATION / "final_optimization_results.csv"
OPTIMIZATION_REPORT = OPTIMIZATION / "final_optimization_report.md"

MODES = ["dense", "sparse", "graph", "hybrid"]
MODE_LABELS = ["Dense", "Sparse", "Graph", "Hybrid"]
COLORS = {
    "dense": "#2F6B9A",
    "sparse": "#E69F00",
    "graph": "#2A9D8F",
    "hybrid": "#7B5BA7",
}
METRIC_COLORS = ["#2F6B9A", "#E69F00", "#2A9D8F", "#7B5BA7"]


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def verify_sources(retrieval: dict, summary_rows: list[dict[str, str]], category_rows: list[dict[str, str]]) -> None:
    """Fail fast when duplicated machine-readable metrics disagree."""
    summary = {row["mode"]: row for row in summary_rows}
    for mode in MODES:
        for metric in ("n", "hit_at_1", "hit_at_3", "hit_at_5", "mrr", "mean_latency_ms", "median_latency_ms"):
            assert close(retrieval["overall"][mode][metric], summary[mode][metric]), (
                f"Metric mismatch: {mode}.{metric}"
            )

    for row in category_rows:
        category, mode = row["category"], row["mode"]
        source = retrieval["by_category"][category][mode]
        for metric in ("n", "hit_at_1", "hit_at_3", "hit_at_5", "mrr"):
            assert close(source[metric], row[metric]), f"Category mismatch: {category}.{mode}.{metric}"

    complete_text = COMPLETE_REPORT.read_text(encoding="utf-8")
    assert "Best Hit@1: Hybrid (0.729)" in complete_text
    assert "Best Hit@3: Sparse (0.875)" in complete_text
    assert "Best Hit@5: Sparse (0.938)" in complete_text
    assert "Best MRR: Hybrid (0.790)" in complete_text

    optimization_text = OPTIMIZATION_REPORT.read_text(encoding="utf-8")
    assert "**KEEP BASELINE**" in optimization_text
    assert "Dense: 0.45" in optimization_text and "Sparse: 0.30" in optimization_text
    assert "Graph: 0.25" in optimization_text and "RRF k: 60" in optimization_text


def style_axis(ax, *, percent: bool = False, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B8C0CC")
    ax.tick_params(colors="#334155", labelsize=10)
    ax.grid(axis=grid_axis, color="#DDE3EA", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")


def finish(fig, filename: str) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(OUTPUT / filename, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax, bars, *, percent: bool = True, fontsize: int = 8) -> None:
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.1%}" if percent else f"{value:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, label, ha="center", va="bottom", fontsize=fontsize, color="#243447")


def retrieval_chart(retrieval: dict) -> None:
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr"]
    labels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    x = np.arange(len(MODES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    for index, (metric, label, color) in enumerate(zip(metrics, labels, METRIC_COLORS)):
        values = [retrieval["overall"][mode][metric] for mode in MODES]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
        add_bar_labels(ax, bars, fontsize=7)
    ax.set_title("Retrieval Method Comparison", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.set_subtitle("48 answerable questions; exact frozen relevance IDs") if hasattr(ax, "set_subtitle") else None
    ax.set_xticks(x, MODE_LABELS)
    ax.set_ylabel("Score")
    style_axis(ax, percent=True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    finish(fig, "retrieval_methods_comparison.png")


def optimization_chart(rows: list[dict[str, str]]) -> None:
    metrics = ["hit_at_1", "hit_at_3", "hit_at_5", "mrr"]
    labels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
    configs = [row["configuration"] for row in rows]
    x = np.arange(len(configs))
    width = 0.19
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    for index, (metric, label, color) in enumerate(zip(metrics, labels, METRIC_COLORS)):
        values = [float(row[metric]) for row in rows]
        ax.bar(x + (index - 1.5) * width, values, width, label=label, color=color)
    ax.axvspan(-0.48, 0.48, color="#E8EDF3", alpha=0.8, zorder=0)
    ax.set_title("Hybrid Weight Optimization Comparison", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.text(0.99, 0.04, "Decision: KEEP BASELINE", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, weight="bold", color="#8A3B12", bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF2E8", "edgecolor": "#F0B789"})
    ax.set_xticks(x, configs)
    ax.set_xlabel("Configuration (Baseline and A–F)")
    ax.set_ylabel("Score")
    style_axis(ax, percent=True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    finish(fig, "hybrid_optimization_comparison.png")


def category_chart(category_rows: list[dict[str, str]]) -> None:
    categories = sorted({row["category"] for row in category_rows})
    lookup = {(row["category"], row["mode"]): float(row["hit_at_3"]) for row in category_rows}
    y = np.arange(len(categories))
    height = 0.19
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    for index, mode in enumerate(MODES):
        values = [lookup[(category, mode)] for category in categories]
        ax.barh(y + (index - 1.5) * height, values, height, label=mode.title(), color=COLORS[mode])
    ax.set_title("Hit@3 by Question Category", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.set_yticks(y, [category.replace("_", " ").title() for category in categories])
    ax.set_xlabel("Hit@3")
    ax.set_xlim(0, 1.05)
    ax.xaxis.set_major_formatter(lambda value, _position: f"{value:.0%}")
    style_axis(ax, grid_axis="x")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    finish(fig, "category_performance.png")


def answer_quality_chart(answer: dict) -> None:
    metrics = [
        ("automatic_acceptable_phrase_hit", "Acceptable phrase hit"),
        ("automatic_evidence_supported_phrase_hit", "Evidence-supported phrase hit"),
        ("refusal_rate", "Refusal rate (2 insufficient-evidence items)"),
    ]
    x = np.arange(len(MODES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    for index, (metric, label) in enumerate(metrics):
        values = [answer["overall"][mode][metric] for mode in MODES]
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=METRIC_COLORS[index])
        add_bar_labels(ax, bars, fontsize=8)
    ax.set_title("Answer-Level Proxy Metrics", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.text(0.5, -0.14, "Deterministic string-match proxies — not semantic correctness or human-rated answer quality", transform=ax.transAxes, ha="center", fontsize=10, color="#5B6472")
    ax.set_xticks(x, MODE_LABELS)
    ax.set_ylabel("Rate")
    style_axis(ax, percent=True)
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=9)
    fig.tight_layout()
    finish(fig, "answer_quality_comparison.png")


def retrieval_latency_chart(retrieval: dict) -> None:
    y = np.arange(len(MODES))
    height = 0.34
    means = [retrieval["overall"][mode]["mean_latency_ms"] for mode in MODES]
    medians = [retrieval["overall"][mode]["median_latency_ms"] for mode in MODES]
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    mean_bars = ax.barh(y - height / 2, means, height, label="Mean", color="#2F6B9A")
    median_bars = ax.barh(y + height / 2, medians, height, label="Median", color="#A7C7E7")
    ax.set_xscale("log")
    ax.set_title("Retrieval Latency Comparison", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.set_yticks(y, MODE_LABELS)
    ax.set_xlabel("Latency (ms, logarithmic scale)")
    style_axis(ax, grid_axis="x")
    for bars in (mean_bars, median_bars):
        for bar in bars:
            ax.text(bar.get_width() * 1.08, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f} ms", va="center", fontsize=9, color="#243447")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    ax.invert_yaxis()
    fig.tight_layout()
    finish(fig, "latency_comparison.png")


def answer_latency_chart(answer: dict) -> None:
    x = np.arange(len(MODES))
    width = 0.34
    generation = [answer["overall"][mode]["mean_generation_latency_ms"] for mode in MODES]
    total = [answer["overall"][mode]["mean_total_latency_ms"] for mode in MODES]
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    bars_a = ax.bar(x - width / 2, generation, width, label="Mean generation", color="#2A9D8F")
    bars_b = ax.bar(x + width / 2, total, width, label="Mean total", color="#7B5BA7")
    ax.set_title("Answer Generation and Total Latency", fontsize=20, weight="bold", color="#172033", pad=18)
    ax.set_xticks(x, MODE_LABELS)
    ax.set_ylabel("Latency (ms)")
    style_axis(ax)
    add_bar_labels(ax, bars_a, percent=False, fontsize=8)
    add_bar_labels(ax, bars_b, percent=False, fontsize=8)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout()
    finish(fig, "answer_latency_comparison.png")


def write_summary(retrieval: dict, answer: dict, optimization_rows: list[dict[str, str]], category_rows: list[dict[str, str]]) -> None:
    baseline = next(row for row in optimization_rows if row["configuration"] == "BASELINE")
    selected = next(row for row in optimization_rows if row["selected_winner"].lower() == "true")
    best_hit1 = max(MODES, key=lambda mode: retrieval["overall"][mode]["hit_at_1"])
    best_hit3 = max(MODES, key=lambda mode: retrieval["overall"][mode]["hit_at_3"])
    best_hit5 = max(MODES, key=lambda mode: retrieval["overall"][mode]["hit_at_5"])
    best_mrr = max(MODES, key=lambda mode: retrieval["overall"][mode]["mrr"])
    report = f"""# Presentation Chart Summary

## retrieval_methods_comparison.png

- Exact sources: `{RETRIEVAL_JSON.relative_to(ROOT)}` (`overall`) and `{RETRIEVAL_CSV.relative_to(ROOT)}`; duplicated values were checked for exact numeric agreement.
- Metrics: Hit@1, Hit@3, Hit@5, and MRR on 48 answerable questions.
- Key finding: {best_hit1.title()} leads Hit@1 ({retrieval['overall'][best_hit1]['hit_at_1']:.3f}) and {best_mrr.title()} leads MRR ({retrieval['overall'][best_mrr]['mrr']:.3f}); {best_hit3.title()} leads Hit@3 ({retrieval['overall'][best_hit3]['hit_at_3']:.3f}) and {best_hit5.title()} leads Hit@5 ({retrieval['overall'][best_hit5]['hit_at_5']:.3f}). Hybrid does not win every metric.
- คำอธิบายสำหรับนำเสนอ: Hybrid เด่นด้านการจัดอันดับผลลัพธ์แรกและ MRR ขณะที่ Sparse ครอบคลุมผลลัพธ์ในอันดับ 3 และ 5 ได้ดีกว่า จึงควรอธิบายว่าแต่ละวิธีมีจุดแข็งต่างกัน

## hybrid_optimization_comparison.png

- Exact sources: `{OPTIMIZATION_CSV.relative_to(ROOT)}` and `{OPTIMIZATION_REPORT.relative_to(ROOT)}`.
- Metrics: Hit@1, Hit@3, Hit@5, and MRR for Baseline and configurations A–F; RRF k=60 for every configuration.
- Key finding: Configuration {selected['configuration']} has higher retrieval metrics than Baseline (MRR {float(selected['mrr']):.3f} vs {float(baseline['mrr']):.3f}), but the recorded final decision is **KEEP BASELINE** because answer-level refusal behavior became worse.
- คำอธิบายสำหรับนำเสนอ: แม้น้ำหนักชุด {selected['configuration']} จะเพิ่มคะแนน retrieval แต่ผลยืนยันระดับคำตอบทำให้ความสามารถในการปฏิเสธคำถามที่ไม่มีหลักฐานลดลง จึงคง baseline 0.45/0.30/0.25

## category_performance.png

- Exact sources: `{CATEGORY_CSV.relative_to(ROOT)}` and `{RETRIEVAL_JSON.relative_to(ROOT)}` (`by_category`); duplicated values were checked for exact numeric agreement.
- Metric: Hit@3 for nine answerable categories. `insufficient_evidence` is excluded because retrieval relevance is N/A.
- Key finding: Graph reaches 1.000 in entity relation, multi-hop relation, comparison, and graph-friendly categories, but scores 0.000 in recommendation and lexical mismatch. Sparse reaches 1.000 in direct fact, comparison, recommendation, and dense-friendly categories.
- คำอธิบายสำหรับนำเสนอ: กราฟช่วยคำถามเชิงความสัมพันธ์ได้ชัดเจน แต่ไม่เหมาะกับทุกหมวด ส่วน Sparse แข็งแรงกับคำถามข้อเท็จจริงและคำแนะนำ จึงเป็นเหตุผลที่ระบบผสมหลาย retriever

## answer_quality_comparison.png

- Exact source: `{ANSWER_JSON.relative_to(ROOT)}` (`overall` and `automatic_metric_limitations`), cross-checked with `{COMPLETE_REPORT.relative_to(ROOT)}`.
- Metrics: acceptable-phrase hit, evidence-supported phrase hit, and refusal rate on two insufficient-evidence questions.
- Key finding: These are deterministic string-match proxies, not semantic correctness or human-rated quality. Graph has the highest acceptable-phrase proxy ({answer['overall']['graph']['automatic_acceptable_phrase_hit']:.1%}) and refusal rate ({answer['overall']['graph']['refusal_rate']:.1%}); Hybrid records {answer['overall']['hybrid']['automatic_acceptable_phrase_hit']:.1%} and {answer['overall']['hybrid']['refusal_rate']:.1%}.
- คำอธิบายสำหรับนำเสนอ: ตัวเลขนี้ใช้ตรวจคำหรือวลีที่กำหนดไว้เท่านั้น จึงใช้เป็นสัญญาณประกอบและยังต้องให้มนุษย์ประเมินคุณภาพภาษาไทยและความถูกต้องเชิงความหมาย

## latency_comparison.png

- Exact sources: `{RETRIEVAL_JSON.relative_to(ROOT)}` (`overall`) and `{RETRIEVAL_CSV.relative_to(ROOT)}`.
- Metrics: mean and median production `retrieve()` latency after one unmeasured warm-up; initialization excluded. A logarithmic x-axis is used because measured values span more than two orders of magnitude.
- Key finding: Graph and Sparse retrieval are sub-millisecond in this run, while Hybrid mean retrieval latency is {retrieval['overall']['hybrid']['mean_latency_ms']:.3f} ms and Dense is {retrieval['overall']['dense']['mean_latency_ms']:.3f} ms.
- คำอธิบายสำหรับนำเสนอ: เวลา retrieval ของทุกวิธียังต่ำเมื่อเทียบกับเวลาสร้างคำตอบ และแกน logarithmic ช่วยให้เห็นค่าที่ต่างกันมากโดยไม่ซ่อน Graph/Sparse

## answer_latency_comparison.png

- Exact source: `{ANSWER_JSON.relative_to(ROOT)}` (`overall`).
- Metrics: mean LLM generation latency and mean end-to-end answer latency; warm-up excluded.
- Key finding: Answer generation dominates total latency. Graph context has the lowest mean total latency ({answer['overall']['graph']['mean_total_latency_ms']:.1f} ms), while Dense has the highest ({answer['overall']['dense']['mean_total_latency_ms']:.1f} ms).
- คำอธิบายสำหรับนำเสนอ: คอขวดหลักอยู่ที่การสร้างคำตอบของ LLM ไม่ใช่ retrieval จึงควรแยกสองสเกลนี้ออกจากกันในการนำเสนอ

## Overall findings

- Strength: Hybrid provides the best Hit@1 and MRR in the production baseline, while Sparse provides the best Hit@3 and Hit@5.
- Strength: Graph is particularly effective for relation-oriented categories.
- Weakness: No single retrieval method dominates every metric or category; Graph is weak for recommendation and lexical mismatch in this dataset.
- Weakness: Answer metrics are proxy checks and subjective Thai answer quality remains `HUMAN_REVIEW_REQUIRED`.
- Decision: **KEEP BASELINE** with Dense=0.45, Sparse=0.30, Graph=0.25, and RRF k=60.
"""
    (OUTPUT / "chart_summary.md").write_text(report, encoding="utf-8")


def validate_pngs() -> None:
    pngs = sorted(OUTPUT.glob("*.png"))
    assert len(pngs) == 6, f"Expected 6 PNG files, found {len(pngs)}"
    for path in pngs:
        assert path.stat().st_size > 10_000, f"PNG too small: {path}"
        with Image.open(path) as image:
            width, height = image.size
            assert width >= 2400 and height >= 1200, f"Image resolution too low: {path} {image.size}"
            assert image.format == "PNG", f"Unexpected image format: {path}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    retrieval = load_json(RETRIEVAL_JSON)
    answer = load_json(ANSWER_JSON)
    summary_rows = load_csv(RETRIEVAL_CSV)
    category_rows = load_csv(CATEGORY_CSV)
    optimization_rows = load_csv(OPTIMIZATION_CSV)

    verify_sources(retrieval, summary_rows, category_rows)
    retrieval_chart(retrieval)
    optimization_chart(optimization_rows)
    category_chart(category_rows)
    answer_quality_chart(answer)
    retrieval_latency_chart(retrieval)
    answer_latency_chart(answer)
    write_summary(retrieval, answer, optimization_rows, category_rows)
    validate_pngs()
    print(f"Generated and validated 6 PNG charts plus chart_summary.md in {OUTPUT}")


if __name__ == "__main__":
    main()
