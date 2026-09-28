# -*- coding: utf-8 -*-
"""
Automated Information Extraction (IE) Pipeline for Songkhla Old Town Knowledge Graph.
Extracts RDF Triples (Subject, Predicate, Object) directly from raw text chunks
using LLM-driven Relation Extraction (Ollama / Local LLM) guided by Domain Ontology.

Eliminates all hardcoded dictionaries and manual Cypher writing.
Generates:
1. finalproject/data/songkhla_knowledge_triples_automated.json
2. finalproject/data/automated_extraction_report.md
"""

import os
import sys
import json
import time
import re
from pathlib import Path
import requests

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "songkhla_rag_chunks.json"
SCHEMA_PATH = DATA_DIR / "songkhla_ontology_schema.json"
OUTPUT_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples_automated.json"
OUTPUT_REPORT_PATH = DATA_DIR / "automated_extraction_report.md"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

# Load Domain Ontology
ONTOLOGY_RELATIONS = [
    "LOCATED_ON",       # Place/FoodShop/Hotel -> Street
    "SERVES",           # FoodShop -> Dish / Drink
    "OFFERS_ACTIVITY",  # Place -> Activity
    "VISITED_ON",       # Place -> TourDay / Time
    "HISTORICAL_ERA",   # Place -> Era / Reign
    "CONNECTS_TO",      # Street -> Street
    "NEARBY"            # Place -> Place
]


def clean_json_response(raw_text: str) -> list:
    """Extracts valid JSON array from LLM response text."""
    # Find JSON array using regex
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    # Try cleaning code blocks
    cleaned = re.sub(r'```json|```', '', raw_text).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def extract_triples_from_chunk(chunk_id: str, title: str, text: str) -> list:
    """Sends raw chunk text to LLM for automated triple extraction."""
    system_prompt = (
        "You are an expert NLP and Knowledge Graph Engineer. Your task is to extract Knowledge Graph Triples "
        "from Thai cultural tourism text based on the given ontology relations. Output strictly a valid JSON array."
    )

    user_prompt = f"""จงทำ Information Extraction สกัดความสัมพันธ์ (Triples: Subject-Predicate-Object) จากข้อความต่อไปนี้:

[หัวข้อ: {title}]
{text}

กฎการสกัด (Ontology Schema Constraints):
1. สกัดเฉพาะข้อเท็จจริงที่ปรากฏในข้อความข้างต้นเท่านั้น ห้ามสร้างข้อมูลเท็จขึ้นมาเอง (Zero Hallucination)
2. กำหนด relation ให้ตรงกับประเภทต่อไปนี้เท่านั้น:
   - LOCATED_ON (สถานที่/ร้านค้า -> ถนนหรือย่าน)
   - SERVES (ร้านอาหาร -> เมนูเด็ด/ของกิน/ของฝาก)
   - OFFERS_ACTIVITY (สถานที่ -> กิจกรรมท่องเที่ยว/ไฮไลท์)
   - VISITED_ON (สถานที่ -> วันหรือเวลาในแผนเที่ยว)
   - HISTORICAL_ERA (สถานที่ -> ยุคสมัยทางประวัติศาสตร์/รัชกาล)
   - CONNECTS_TO (ถนน/เส้นทาง -> ถนนหรือตรอกที่เชื่อมถึงกัน)
   - NEARBY (สถานที่ -> สถานที่ใกล้เคียง)
3. ส่งคืนผลลัพธ์เป็น JSON Array เท่านั้น ในรูปแบบ:
[
  {{
    "subject": "ชื่อเอนทิตี้นามต้นทาง",
    "relation": "ชื่อความสัมพันธ์ตามกฎข้อ 2",
    "object": "ชื่อเอนทิตี้นามปลายทาง",
    "evidence": "ข้อความสั้นๆ ในเนื้อหาที่ยืนยันความสัมพันธ์นี้"
  }}
]"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8
        }
    }

    try:
        t0 = time.time()
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        elapsed = time.time() - t0
        
        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "")
            triples = clean_json_response(content)
            # Add metadata
            for t in triples:
                t["source_chunk"] = chunk_id
                t["chunk_title"] = title
            return triples, elapsed, None
        else:
            return [], 0.0, f"HTTP Error {res.status_code}"
    except Exception as e:
        return [], 0.0, str(e)


def run_automated_extraction():
    print("=" * 70)
    print("🚀 STARTING AUTOMATED LLM KNOWLEDGE GRAPH EXTRACTION PIPELINE")
    print(f"Model: {MODEL_NAME} | Offline Local Extraction")
    print("=" * 70)

    if not CHUNKS_PATH.exists():
        print(f"Error: Chunks file not found at {CHUNKS_PATH}")
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    all_triples = []
    extraction_logs = []
    start_total_time = time.time()

    print(f"Processing {len(chunks)} raw text chunks...\n")

    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", f"chunk_{i:03d}")
        title = c.get("title", "")
        content = c.get("content", "")

        print(f"[{i}/{len(chunks)}] Extracting from {cid}: '{title[:40]}...' ", end="", flush=True)
        triples, latency, err = extract_triples_from_chunk(cid, title, content)

        if err:
            print(f"❌ Error: {err}")
            extraction_logs.append({
                "chunk_id": cid, "title": title, "triples_count": 0, "latency_s": 0.0, "status": err
            })
        else:
            print(f"✅ {len(triples)} triples extracted in {latency:.2f}s")
            all_triples.extend(triples)
            extraction_logs.append({
                "chunk_id": cid, "title": title, "triples_count": len(triples), "latency_s": round(latency, 2), "status": "Success"
            })

    total_time = time.time() - start_total_time

    # Save Triples JSON
    with open(OUTPUT_TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 Saved {len(all_triples)} automated triples to {OUTPUT_TRIPLES_PATH}")

    # Relation breakdown
    relation_counts = {}
    for t in all_triples:
        rel = t.get("relation", "UNKNOWN")
        relation_counts[rel] = relation_counts.get(rel, 0) + 1

    # Generate Markdown Audit Report
    report = f"""# 📊 Automated Knowledge Graph Extraction Audit Report
**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Extraction Engine:** Local LLM `{MODEL_NAME}` (Ollama GPU)  
**Total Raw Chunks Processed:** {len(chunks)} chunks  
**Total Triples Automatically Extracted:** {len(all_triples)} triples  
**Total Extraction Time:** {total_time:.2f} seconds (Average {total_time/len(chunks):.2f}s per chunk)  

---

## 1. การกระจายตัวตามประเภทความสัมพันธ์ (Ontology Relation Distribution)

| ความสัมพันธ์ (Relation) | จำนวน Triples ที่สกัดได้ | สัดส่วน (%) | คำอธิบายความหมายเชิงโครงสร้าง |
| :--- | :---: | :---: | :--- |
"""
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(all_triples)) * 100 if all_triples else 0
        report += f"| `{rel}` | **{count}** | {pct:.1f}% | ความสัมพันธ์ที่สกัดได้จากข้อความจริง |\n"

    report += """
---

## 2. ตัวอย่าง Triples ที่สกัดได้จริงจากข้อความ (Sample Extracted Triples with Text Evidence)

| Chunk ต้นทาง | Subject (ประธาน) | Relation (กริยา) | Object (กรรม) | Evidence (หลักฐานในข้อความจริง) |
| :--- | :--- | :---: | :--- | :--- |
"""
    for t in all_triples[:15]:
        report += f"| `{t.get('source_chunk', '')}` | **{t.get('subject', '')}** | `{t.get('relation', '')}` | **{t.get('object', '')}** | *\"{t.get('evidence', '')}\"* |\n"

    report += """
---

## 3. บันทึกผลการสกัดราย Chunk (Chunk-by-Chunk Extraction Log)

| Chunk ID | หัวข้อเอกสาร | จำนวน Triples | เวลาประมวลผล (s) | สถานะ |
| :--- | :--- | :---: | :---: | :---: |
"""
    for log in extraction_logs:
        report += f"| `{log['chunk_id']}` | {log['title']} | **{log['triples_count']}** | {log['latency_s']}s | {log['status']} |\n"

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Saved detailed audit report to {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    run_automated_extraction()
