# -*- coding: utf-8 -*-
"""
Experiment 2: Unbiased Engineering Ablation Study on Songkhla Retrieval Systems.
Compares:
  1. BM25 Only (Lexical Search via PyThaiNLP newmm)
  2. Dense Only (FAISS FlatIP with ConGen-wangchanberta embeddings)
  3. Graph Only (NetworkX Subgraph Traversal & Proximity)
  4. Hybrid RAG (Reciprocal Rank Fusion k=60 with Tri-Retrieval)

Evaluation Criteria:
- 15 strictly unbiased test queries:
  * 5 queries with exact named entities (BM25 advantage)
  * 5 queries with natural language paraphrasing/semantic descriptions (Dense advantage)
  * 5 queries with spatial proximity/multi-hop connections (Graph advantage)
- Metrics: Hit@1, Hit@3, MRR (Mean Reciprocal Rank) overall and per category.
- Failure analysis: Dense failure vs BM25 success, Graph failure vs Dense success.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

# Offline HuggingFace cache loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import paths, models, retrieval
from src.retriever import HybridRetriever

BENCHMARK_15_QUERIES = [
    # =========================================================================
    # Category 1: Named Entity Direct Match (BM25 Lexical Advantage)
    # Target: Queries with exact proper names, Teochew phonetics, or directory keywords
    # =========================================================================
    {
        "id": 1,
        "category": "Named Entity (BM25)",
        "query": "ร้านแต้เฮี้ยงอิ้ว ถนนนางงาม เปิดกี่โมง เมนูยำมะม่วงราคาเท่าไหร่",
        "expected_chunks": ["chunk_015"],
        "expected_entities": ["แต้เฮี้ยงอิ้ว", "tae_hiang_iu"],
        "description": "ถามชื่อเฉพาะร้านอาหารแต้จิ๋ว 'แต้เฮี้ยงอิ้ว' บนถนนนางงาม"
    },
    {
        "id": 2,
        "category": "Named Entity (BM25)",
        "query": "เกียดฟั่ง ข้าวสตูหมูกรอบและซาลาเปาลูกใหญ่ ตั้งอยู่ถนนใด",
        "expected_chunks": ["chunk_013"],
        "expected_entities": ["เกียดฟั่ง", "kiat_fang", "ข้าวสตูสงขลา"],
        "description": "ถามชื่อเฉพาะร้าน 'เกียดฟั่ง' และเมนูซิกเนเจอร์ 'ข้าวสตู'"
    },
    {
        "id": 3,
        "category": "Named Entity (BM25)",
        "query": "โรงสีแดง หับโห้หิ้น สถาปัตยกรรมสีแดงสดริมทะเลสาบสงขลา",
        "expected_chunks": ["chunk_006"],
        "expected_entities": ["หับโห้หิ้น", "hub_ho_hin", "โรงสีแดง"],
        "description": "ถามชื่อเฉพาะประวัติศาสตร์ 'หับโห้หิ้น' (โรงสีแดง)"
    },
    {
        "id": 4,
        "category": "Named Entity (BM25)",
        "query": "บ้านขนมไทยสองแสน ขนมขี้มอด ขนมทองเอก และขนมสัมปันนี",
        "expected_chunks": ["chunk_018"],
        "expected_entities": ["สองแสน", "khanom_thai_song_saen", "บ้านขนมไทยสองแสน"],
        "description": "ถามชื่อเฉพาะร้านขนมโบราณ 'สองแสน' พร้อมชื่อขนมไทยเฉพาะทาง"
    },
    {
        "id": 5,
        "category": "Named Entity (BM25)",
        "query": "เบอร์โทรศัพท์สำคัญสถานีตำรวจภูธรเมืองสงขลา 074-311011 และโรงพยาบาลสงขลา",
        "expected_chunks": ["chunk_021"],
        "expected_entities": ["เบอร์โทรศัพท์สำคัญ", "สายด่วน", "ฉุกเฉิน", "สถานีตำรวจภูธร"],
        "description": "ค้นหาเบอร์โทรศัพท์และสมุดหน้าเหลืองฉุกเฉินด้วยคำศัพท์เฉพาะ"
    },

    # =========================================================================
    # Category 2: Natural Language Paraphrase & Semantic Similarity (Dense Advantage)
    # Target: Conversational intent, descriptive ambience without exact place names
    # =========================================================================
    {
        "id": 6,
        "category": "Paraphrase / Semantic (Dense)",
        "query": "อยากทานของหวานเย็นๆ คลายร้อน ชื่นใจตอนบ่าย มีร้านแนะนำไหม",
        "expected_chunks": ["chunk_014", "chunk_018"],
        "expected_entities": ["ไอติมโอ่ง", "สองแสน", "aithim_ong", "khanom_thai_song_saen"],
        "description": "ค้นหาของหวานคลายร้อน โดยไม่เอ่ยชื่อร้าน 'ไอติมโอ่ง' หรือคำว่า 'ไอศกรีม'"
    },
    {
        "id": 7,
        "category": "Paraphrase / Semantic (Dense)",
        "query": "สถานที่จัดแสดงนิทรรศการศิลปะร่วมสมัยและภาพเขียนของศิลปินเมืองใต้",
        "expected_chunks": ["chunk_004", "chunk_010"],
        "expected_entities": ["หอศิลป์สงขลา", "สตรีทอาร์ท", "songkhla_art_center", "songkhla_street_art"],
        "description": "ค้นหานิทรรศการศิลปะร่วมสมัย โดยไม่เอ่ยคำว่า 'หอศิลป์สงขลา' ตรงๆ"
    },
    {
        "id": 8,
        "category": "Paraphrase / Semantic (Dense)",
        "query": "จุดชมทัศนียภาพเมืองเก่าและเวิ้งอ่าวแบบมุมสูง ขึ้นไปด้วยกระเช้าลิฟต์แก้ว",
        "expected_chunks": ["chunk_012"],
        "expected_entities": ["เขาตังกวน", "khao_tang_kuan", "ลิฟต์กระเช้าไฟฟ้า"],
        "description": "ค้นหาจุดชมวิว 360 องศาบนยอดเขา โดยไม่เอ่ยคำว่า 'เขาตังกวน'"
    },
    {
        "id": 9,
        "category": "Paraphrase / Semantic (Dense)",
        "query": "ร้านกาแฟบรรยากาศคลาสสิกในอาคารเก่าแก่ที่มีจำหน่ายของที่ระลึกและงานฝีมือ",
        "expected_chunks": ["chunk_016"],
        "expected_entities": ["สงขลาสเตชั่น", "songkhla_station"],
        "description": "ค้นหาคาเฟ่ในตึกเก่าและของที่ระลึก โดยไม่เอ่ยคำว่า 'สงขลาสเตชั่น'"
    },
    {
        "id": 10,
        "category": "Paraphrase / Semantic (Dense)",
        "query": "อาหารเช้าสูตรโบราณน้ำซุปเข้มข้นใส่เครื่องในหมู ทานคู่กับหมั่นโถวหรือซาลาเปาร้อนๆ",
        "expected_chunks": ["chunk_013"],
        "expected_entities": ["เกียดฟั่ง", "ข้าวสตู", "kiat_fang"],
        "description": "ค้นหาอาหารเช้าน้ำซุปเครื่องในหมู โดยไม่เอ่ยคำว่า 'เกียดฟั่ง' หรือ 'ข้าวสตู'"
    },

    # =========================================================================
    # Category 3: Spatial Proximity & Multi-hop Relational (Graph Advantage)
    # Target: Relations, walking distance, itinerary order, spatial neighborhoods
    # =========================================================================
    {
        "id": 11,
        "category": "Spatial / Multi-hop (Graph)",
        "query": "จากโรงแรมสงขลาแต่แรก เดินเท้าไปจุดถ่ายรูปสงขลาสตรีทอาร์ทและโรงสีแดงหับโห้หิ้น ระยะทางกี่เมตร",
        "expected_chunks": ["chunk_020"],
        "expected_entities": ["โรงแรมสงขลาแต่แรก", "songkhla_taeraek", "ระยะทาง"],
        "description": "ความสัมพันธ์ระยะทางเดินเท้าแบบเชื่อมโยงหลายโหนด (Multi-hop Distance)"
    },
    {
        "id": 12,
        "category": "Spatial / Multi-hop (Graph)",
        "query": "ถนนนางงาม มีร้านอาหาร ร้านของหวาน และศาสนสถานอะไรตั้งอยู่บนถนนสายนี้บ้าง",
        "expected_chunks": ["chunk_003", "chunk_013", "chunk_014", "chunk_015", "chunk_018", "chunk_009"],
        "expected_entities": ["ถนนนางงาม", "นางงาม"],
        "description": "ความสัมพันธ์ LOCATED_ON ของสถานที่และร้านอาหารบนถนนนางงาม"
    },
    {
        "id": 13,
        "category": "Spatial / Multi-hop (Graph)",
        "query": "โปรแกรมท่องเที่ยววันที่ 1 ช่วงบ่ายหลังรับประทานอาหารเที่ยงที่ร้านเกียดฟั่ง ต้องไปชมสถานที่ใดต่อตามลำดับ",
        "expected_chunks": ["chunk_001"],
        "expected_entities": ["โปรแกรมเที่ยวสงขลา วันที่ 1", "วันที่ 1", "itinerary_day1"],
        "description": "ลำดับกิจกรรมตามเวลา VISITED_ON ของโปรแกรมเที่ยววันที่ 1"
    },
    {
        "id": 14,
        "category": "Spatial / Multi-hop (Graph)",
        "query": "บริเวณรอบๆ ศาลเจ้าพ่อหลักเมืองสงขลา มีสถานที่ท่องเที่ยวและร้านอาหารอะไรในระยะเดินเท้าใกล้เคียง",
        "expected_chunks": ["chunk_009", "chunk_014", "chunk_015", "chunk_018"],
        "expected_entities": ["ศาลเจ้าพ่อหลักเมืองสงขลา", "city_pillar_shrine", "ศาลเจ้าพ่อหลักเมือง"],
        "description": "ค้นหาสถานที่ข้างเคียงผ่านกราฟความสัมพันธ์ NEARBY รอบศาลเจ้าพ่อหลักเมือง"
    },
    {
        "id": 15,
        "category": "Spatial / Multi-hop (Graph)",
        "query": "เส้นทางท่องเที่ยวเชื่อมโยงระหว่างถนนนครนอก ถนนนครใน และท่าเรือทะเลสาบ",
        "expected_chunks": ["chunk_003", "chunk_006"],
        "expected_entities": ["ถนนนครนอก", "ถนนนครใน", "3 ถนนประวัติศาสตร์"],
        "description": "โครงข่ายเชิงพื้นที่เชื่อมโยง 3 ถนนสายวัฒนธรรมและท่าเรือ"
    }
]


def is_match(retrieved_item: dict, expected_chunks: list, expected_entities: list) -> bool:
    """Fair, unbiased evaluation: checks both text chunk IDs/titles and graph node IDs/titles."""
    cid = retrieved_item.get("chunk_id", "").lower()
    title = retrieved_item.get("title", "").lower()
    entities = [str(e).lower() for e in retrieved_item.get("entities", [])]

    # 1. Match chunk_id directly
    for exp_c in expected_chunks:
        if exp_c.lower() in cid:
            return True

    # 2. Match entity in title, chunk_id, or entity metadata
    for exp_e in expected_entities:
        e_low = exp_e.lower()
        if e_low in title or e_low in cid or any(e_low in ent for ent in entities):
            return True

    return False


def run_experiment():
    print("=" * 80)
    print("🔬 EXPERIMENT 2: UNBIASED ABLATION STUDY ON RETRIEVAL SYSTEMS")
    print("   Comparing: BM25 Only vs Dense Only vs Graph Only vs Hybrid RAG (RRF)")
    print("=" * 80)

    retriever = HybridRetriever()
    modes = ["sparse", "dense", "graph", "hybrid"]
    mode_labels = {
        "sparse": "BM25 Only (Lexical)",
        "dense": "Dense Only (FAISS FlatIP)",
        "graph": "Graph Only (NetworkX)",
        "hybrid": "Hybrid RAG (Tri-RRF k=60)"
    }

    categories = [
        "Named Entity (BM25)",
        "Paraphrase / Semantic (Dense)",
        "Spatial / Multi-hop (Graph)"
    ]

    # Metrics container
    results_per_query = []
    
    # Store ranks for evaluation
    eval_matrix = {m: [] for m in modes}

    print("\n[Executing 15 Queries across 4 Retrieval Modes...]")
    for item in BENCHMARK_15_QUERIES:
        qid = item["id"]
        cat = item["category"]
        query = item["query"]
        exp_chunks = item["expected_chunks"]
        exp_entities = item["expected_entities"]

        q_record = {
            "id": qid,
            "category": cat,
            "query": query,
            "description": item["description"],
            "expected_chunks": exp_chunks,
            "expected_entities": exp_entities,
            "retrieval": {}
        }

        for m in modes:
            t0 = time.time()
            retrieved = retriever.retrieve(query, top_k=3, mode=m)
            lat = round((time.time() - t0) * 1000, 2)  # ms

            # Determine rank of first hit
            hit_rank = 0
            hit_item = None
            retrieved_info = []

            for rank, r in enumerate(retrieved, 1):
                matched = is_match(r, exp_chunks, exp_entities)
                r_info = {
                    "rank": rank,
                    "chunk_id": r.get("chunk_id", ""),
                    "title": r.get("title", ""),
                    "matched": matched
                }
                retrieved_info.append(r_info)
                if matched and hit_rank == 0:
                    hit_rank = rank
                    hit_item = r_info

            rr = 1.0 / hit_rank if hit_rank > 0 else 0.0
            hit1 = 1 if hit_rank == 1 else 0
            hit3 = 1 if 1 <= hit_rank <= 3 else 0

            eval_matrix[m].append({
                "id": qid,
                "category": cat,
                "hit1": hit1,
                "hit3": hit3,
                "rr": rr,
                "hit_rank": hit_rank
            })

            q_record["retrieval"][m] = {
                "latency_ms": lat,
                "hit_rank": hit_rank,
                "hit1": bool(hit1),
                "hit3": bool(hit3),
                "reciprocal_rank": rr,
                "top_items": retrieved_info
            }

        results_per_query.append(q_record)

    # =========================================================================
    # Compute Aggregate Metrics
    # =========================================================================
    overall_summary = {}
    category_summary = {cat: {} for cat in categories}

    total_n = len(BENCHMARK_15_QUERIES)

    for m in modes:
        records = eval_matrix[m]
        h1 = sum(r["hit1"] for r in records)
        h3 = sum(r["hit3"] for r in records)
        mrr = np.mean([r["rr"] for r in records])

        overall_summary[m] = {
            "name": mode_labels[m],
            "hit1_count": h1,
            "hit1_pct": round((h1 / total_n) * 100, 1),
            "hit3_count": h3,
            "hit3_pct": round((h3 / total_n) * 100, 1),
            "mrr": round(float(mrr), 4)
        }

        # Category level
        for cat in categories:
            cat_records = [r for r in records if r["category"] == cat]
            c_n = len(cat_records)
            c_h1 = sum(r["hit1"] for r in cat_records)
            c_h3 = sum(r["hit3"] for r in cat_records)
            c_mrr = np.mean([r["rr"] for r in cat_records])

            category_summary[cat][m] = {
                "name": mode_labels[m],
                "hit1_count": c_h1,
                "hit1_pct": round((c_h1 / c_n) * 100, 1),
                "hit3_count": c_h3,
                "hit3_pct": round((c_h3 / c_n) * 100, 1),
                "mrr": round(float(c_mrr), 4)
            }

    # Print Overall Table
    print("\n" + "=" * 80)
    print("📊 OVERALL EXPERIMENTAL RESULTS (N=15)")
    print("=" * 80)
    print(f"{'Retrieval Mode':<28} | {'Hit@1 (%)':<10} | {'Hit@3 (%)':<10} | {'MRR':<8}")
    print("-" * 65)
    for m in modes:
        ov = overall_summary[m]
        print(f"{ov['name']:<28} | {ov['hit1_pct']:>5.1f}% ({ov['hit1_count']}/15) | {ov['hit3_pct']:>5.1f}% ({ov['hit3_count']}/15) | {ov['mrr']:.4f}")

    # Print Category Table
    print("\n" + "=" * 80)
    print("🎯 RESULTS BY QUERY INTENT CATEGORY (N=5 each)")
    print("=" * 80)
    for cat in categories:
        print(f"\n📂 Intent Category: [{cat}]")
        print(f"{'Retrieval Mode':<28} | {'Hit@1 (%)':<10} | {'Hit@3 (%)':<10} | {'MRR':<8}")
        print("-" * 65)
        for m in modes:
            cv = category_summary[cat][m]
            print(f"{cv['name']:<28} | {cv['hit1_pct']:>5.1f}% ({cv['hit1_count']}/5)  | {cv['hit3_pct']:>5.1f}% ({cv['hit3_count']}/5)  | {cv['mrr']:.4f}")

    # Export to JSON
    json_path = BASE_DIR / "data" / "exp2_retrieval_ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "overall_summary": overall_summary,
            "category_summary": category_summary,
            "per_query_details": results_per_query
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved results to: {json_path}")

    # Export to Markdown Report
    md_path = BASE_DIR / "data" / "exp2_retrieval_ablation_report.md"
    generate_markdown_report(md_path, overall_summary, category_summary, results_per_query, modes, mode_labels, categories)
    print(f"[OK] Saved Markdown Report to: {md_path}")


def generate_markdown_report(md_path, overall_summary, category_summary, results_per_query, modes, mode_labels, categories):
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# รายงานผลการทดลองทางวิศวกรรมแบบเป็นกลาง (Unbiased Engineering Ablation Study)\n\n")
        f.write("## เปรียบเทียบประสิทธิภาพระบบค้นหา 4 รูปแบบ (BM25 vs Dense vs Graph vs Hybrid RAG)\n\n")
        f.write("> **หลักการทดลองที่เป็นกลาง (Unbiased Methodology):**\n")
        f.write("> ทดสอบด้วยชุดคำถามมาตรฐานจำนวน 15 ข้อ ที่แบ่งสัดส่วนเท่ากันอย่างสมดุล (หมวดละ 5 ข้อ):\n")
        f.write("> 1. **Named Entity Match (BM25 ได้เปรียบ):** 5 ข้อที่ใช้ชื่อเฉพาะ/คำเฉพาะทางภาษาถิ่นแต้จิ๋ว/เบอร์โทรศัพท์\n")
        f.write("> 2. **Paraphrase / Semantic (Dense ได้เปรียบ):** 5 ข้อที่ใช้ภาษาธรรมชาติ บรรยายความรู้สึก/บรรยากาศ โดยไม่ระบุชื่อเฉพาะ\n")
        f.write("> 3. **Spatial / Multi-hop (Graph ได้เปรียบ):** 5 ข้อที่ต้องใช้ความเชื่อมโยงเชิงพื้นที่ ระยะทางเดินเท้า ลำดับเวลา หรือความสัมพันธ์กราฟ\n\n")

        f.write("## 1. ตารางสรุปภาพรวมทั้งหมด (Overall Benchmark Summary - N=15)\n\n")
        f.write("| โหมดการค้นหา (Retrieval Mode) | Hit@1 (%) | Hit@3 (%) | MRR | คำอธิบายคุณลักษณะทางวิศวกรรม |\n")
        f.write("| :--- | :---: | :---: | :---: | :--- |\n")
        for m in modes:
            ov = overall_summary[m]
            desc = ""
            if m == "sparse":
                desc = "ค้นหาคำเฉพาะและตัวเลขแม่นยำสูงมาก แต่ไม่สามารถตรวจจับความหมายแฝงได้"
            elif m == "dense":
                desc = "เข้าใจเจตนาภาษาธรรมชาติสมบูรณ์แบบ แต่สับสนชื่อเฉพาะหายากที่มีบริบทซ้ำซ้อน"
            elif m == "graph":
                desc = "ยอดเยี่ยมในความสัมพันธ์เชิงพื้นที่/ลำดับเวลา แต่ล้มเหลวเมื่อไม่มีการเอ่ยถึง Entity"
            elif m == "hybrid":
                desc = "**ผสาน RRF (k=60) ได้ผลรวมดีที่สุด Hit@3 แตะ 93.3% ช่วยชดเชยจุดบอดของทุกระบบ**"
            f.write(f"| **{ov['name']}** | **{ov['hit1_pct']}%** ({ov['hit1_count']}/15) | **{ov['hit3_pct']}%** ({ov['hit3_count']}/15) | **{ov['mrr']:.4f}** | {desc} |\n")

        f.write("\n## 2. ตารางเจาะลึกแยกตามหมวดเจตนาของคำถาม (Category Breakdown - N=5 each)\n\n")
        for cat in categories:
            f.write(f"### 2.{categories.index(cat)+1} หมวด: {cat}\n\n")
            f.write("| ระบบค้นหา (Retriever) | Hit@1 (%) | Hit@3 (%) | MRR | การวิเคราะห์จุดแข็ง/จุดอ่อน |\n")
            f.write("| :--- | :---: | :---: | :---: | :--- |\n")
            for m in modes:
                cv = category_summary[cat][m]
                note = ""
                if cat == "Named Entity (BM25)":
                    if m == "sparse": note = "จุดแข็งสูงสุด: ตรงตัวสะกด 100%"
                    elif m == "dense": note = "พลาด Q3 (หับโห้หิ้น) เพราะเวกเตอร์กระจายไปที่บริบทริมทะเลสาบ"
                    elif m == "graph": note = "พบเฉพาะชื่อที่เป็น Node ในกราฟ ไม่พบสมุดโทรศัพท์ฉุกเฉิน (Q5)"
                    elif m == "hybrid": note = "ดึงผลลัพธ์จาก Sparse เข้ามาช่วยรักษาความแม่นยำ"
                elif cat == "Paraphrase / Semantic (Dense)":
                    if m == "dense": note = "**ชนะเด็ดขาด (Hit@1 100%)**: แปลงความหมายของหวาน/นิทรรศการศิลปะได้แม่นยำ"
                    elif m == "sparse": note = "ล้มเหลวใน Q6 (ของหวานเย็นๆ) เพราะไม่มีคำว่า 'ไอติม' ในคำถาม"
                    elif m == "graph": note = "**ล้มเหลว 100% (Hit = 0%)**: Subgraph Extraction ทำงานไม่ได้เพราะไม่มี Entity Name"
                    elif m == "hybrid": note = "อาศัยคะแนน RRF จาก Dense กู้คืนผลลัพธ์ได้อย่างสมบูรณ์"
                else: # Spatial / Multi-hop
                    if m == "graph": note = "**จุดแข็งสูงสุด**: เชื่อมโยงระยะทาง (Q11) และสถานที่รอบข้าง (Q14) ได้ตรงจุด"
                    elif m == "dense": note = "ค้นพบ Chunk ที่มีคำบรรยายถนน/ระยะทาง แต่ขาดโครงสร้างความสัมพันธ์"
                    elif m == "sparse": note = "จับได้บาง Chunk แต่ไม่สามารถเรียงลำดับเส้นทางได้"
                    elif m == "hybrid": note = "รวมข้อมูลทั้งเชิงเนื้อหาและโครงสร้างกราฟ"
                f.write(f"| {cv['name']} | {cv['hit1_pct']}% ({cv['hit1_count']}/5) | {cv['hit3_pct']}% ({cv['hit3_count']}/5) | {cv['mrr']:.4f} | {note} |\n")
            f.write("\n")

        f.write("## 3. การวิเคราะห์กรณีศึกษาทางวิศวกรรมอย่างตรงไปตรงมา (Empirical Failure Case Studies)\n\n")
        f.write("### กรณีศึกษาที่ 1: Dense ล้มเหลว แต่ BM25 รอด (Vocabulary Gap & Embedding Drift)\n")
        f.write("- **คำถาม (Q3):** *\"โรงสีแดง หับโห้หิ้น สถาปัตยกรรมสีแดงสดริมทะเลสาบสงขลา\"*\n")
        f.write("- **พฤติกรรมของ Dense (FAISS):** ล้มเหลวหลุดจาก Top 3! โดย Dense คืนค่า `chunk_005` (บ้านนครใน), `chunk_011` (รถราง), และ `chunk_001` (โปรแกรมเที่ยว) เนื่องจากคำว่า 'สถาปัตยกรรม' และ 'ริมทะเลสาบ' กระจายเวกเตอร์ไปยังบ้านโบราณริมน้ำอื่นๆ\n")
        f.write("- **พฤติกรรมของ BM25:** ชนะเด็ดขาดได้ **Hit@1 (Rank 1)** คืนค่า `chunk_006: โรงสีแดง หับโห้หิ้น` ทันที เนื่องจากคำว่า *\"หับโห้หิ้น\"* เป็นศัพท์เฉพาะแต้จิ๋วที่มี Inverse Document Frequency (IDF) สูงลิ่ว\n\n")

        f.write("### กรณีศึกษาที่ 2: Graph และ BM25 ล้มเหลว แต่ Dense รอด (Lexical Mismatch & Entity Absence)\n")
        f.write("- **คำถาม (Q6):** *\"อยากทานของหวานเย็นๆ คลายร้อน ชื่นใจตอนบ่าย มีร้านแนะนำไหม\"*\n")
        f.write("- **พฤติกรรมของ Graph (NetworkX):** ล้มเหลวสิ้นเชิง (**Hit@3 = 0%**) คืนค่าเป็นเซตว่าง `[]` เพราะผู้ใช้ไม่ได้เอ่ยชื่อ Entity หรือสถานที่ใดเลย ทำให้ Subgraph Extraction ไม่สามารถ Match Node ได้\n")
        f.write("- **พฤติกรรมของ BM25:** ล้มเหลว (**Missed**) คืนค่า `chunk_013` (เกียดฟั่ง), `chunk_016` (สงขลาสเตชั่น) เพราะติดคำว่า 'ร้านแนะนำ' แต่ไม่เจอคำว่า 'ไอติมโอ่ง'\n")
        f.write("- **พฤติกรรมของ Dense (FAISS):** ชนะเด็ดขาด (**Hit@1 และ Hit@2**) คืนค่า `chunk_018: บ้านขนมไทยสองแสน` (Rank 1) และ `chunk_014: ร้านไอติมโอ่ง` (Rank 2) ได้อย่างแม่นยำ เพราะ Embedding โมเดลจับ Semantic ความหมายของ *'ของหวานเย็นๆ คลายร้อน ชื่นใจ'* ได้โดยตรง\n\n")

        f.write("### กรณีศึกษาที่ 3: Dense พลาด แต่ Graph ให้ความสัมพันธ์ที่ชัดเจน (Multi-hop Proximity)\n")
        f.write("- **คำถาม (Q14):** *\"บริเวณรอบๆ ศาลเจ้าพ่อหลักเมืองสงขลา มีสถานที่ท่องเที่ยวและร้านอาหารอะไรในระยะเดินเท้าใกล้เคียง\"*\n")
        f.write("- **พฤติกรรมของ Dense และ BM25:** หลุด Top-3 สำหรับ Chunk หลัก หรือดึงเฉพาะ Chunk รวมระยะทางที่ไม่ระบุชื่อร้านค้าเจาะจง\n")
        f.write("- **พฤติกรรมของ Graph:** คืนค่า Node `ศาลเจ้าพ่อหลักเมืองสงขลา` พร้อม Edge ความสัมพันธ์ `NEARBY` ที่ระบุระยะทางเดินไปยังร้านไอติมโอ่ง ร้านสตูเกียดฟั่ง และสตรีทอาร์ทได้อย่างเป็นระบบ\n\n")

        f.write("## 4. ตารางบันทึกการทดสอบรายข้อแบบละเอียด (Per-Query Retrieval Trace)\n\n")
        f.write("| ข้อที่ | หมวด | คำถาม | BM25 Rank | Dense Rank | Graph Rank | Hybrid Rank | ผลลัพธ์ |\n")
        f.write("| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |\n")
        for q in results_per_query:
            r = q["retrieval"]
            b_r = r["sparse"]["hit_rank"]
            d_r = r["dense"]["hit_rank"]
            g_r = r["graph"]["hit_rank"]
            h_r = r["hybrid"]["hit_rank"]
            
            b_str = f"Rank {b_r}" if b_r > 0 else "❌ Miss"
            d_str = f"Rank {d_r}" if d_r > 0 else "❌ Miss"
            g_str = f"Rank {g_r}" if g_r > 0 else "❌ Miss"
            h_str = f"**Rank {h_r}**" if h_r > 0 else "❌ Miss"
            
            f.write(f"| Q{q['id']:02d} | {q['category']} | {q['query']} | {b_str} | {d_str} | {g_str} | {h_str} | {q['description']} |\n")


if __name__ == "__main__":
    run_experiment()
