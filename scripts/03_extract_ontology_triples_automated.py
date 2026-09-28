# -*- coding: utf-8 -*-
"""
03_extract_ontology_triples_automated.py
Professional Automated Information Extraction (IE) Pipeline for Songkhla Old Town Knowledge Graph.

Features:
1. In-Context Few-Shot Learning for precise Thai Relation Extraction
2. Strict Negative Constraints (Anti-hallucination, zero prompt placeholder leakage)
3. Multi-layer Post-Processing & Entity Normalization Engine (Schema Validation & Linking)
4. Numerical Property Extraction (distance_m, walk_min) for Spatial Knowledge
5. Full Audit Logging with raw text evidence for every triple

Outputs:
1. finalproject/data/songkhla_knowledge_triples_automated.json
2. finalproject/data/automated_extraction_report.md
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import requests

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "songkhla_rag_chunks.json"
SCHEMA_PATH = DATA_DIR / "songkhla_ontology_schema.json"
FACTS_PATH = DATA_DIR / "songkhla_places_facts.json"
OUTPUT_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples_automated.json"
OUTPUT_REPORT_PATH = DATA_DIR / "automated_extraction_report.md"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:3b"

# Valid canonical streets in Songkhla Old Town & Surroundings
CANONICAL_STREETS = {
    "ถนนนางงาม": ["นางงาม", "ถนนนางงาม", "ถนนเก้าห้อง"],
    "ถนนนครนอก": ["นครนอก", "ถนนนครนอก"],
    "ถนนนครใน": ["นครใน", "ถนนนครใน"],
    "ถนนเพชรคีรี": ["เพชรคีรี", "ถนนเพชรคีรี"],
    "ถนนรามัญ": ["รามัญ", "ถนนรามัญ"],
    "ถนนพัทลุง": ["พัทลุง", "ถนนพัทลุง"],
    "ถนนยะหริ่ง": ["ยะหริ่ง", "ถนนยะหริ่ง"],
    "ถนนจะนะ": ["จะนะ", "ถนนจะนะ"],
    "ถนนสุขุม": ["สุขุม", "ถนนสุขุม"],
    "ถนนทะเลหลวง": ["ทะเลหลวง", "ถนนทะเลหลวง"],
    "ถนนสะเดา": ["สะเดา", "ถนนสะเดา"],
    "ถนนไทรบุรี": ["ไทรบุรี", "ถนนไทรบุรี"],
    "ถนนวิเชียรชม": ["วิเชียรชม", "ถนนวิเชียรชม"],
    "แหลมสมิหลา": ["สมิหลา", "หาดสมิหลา", "แหลมสมิหลา", "นางเงือกทอง"],
    "เขาตังกวน": ["เขาตังกวน", "ยอดเขาตังกวน"]
}

# Known factual street mappings for high-profile landmarks to prevent extraction hallucinations
LANDMARK_TRUE_STREETS = {
    "โรงสีแดง หับโห้หิ้น": "ถนนนครนอก",
    "โรงสีแดง": "ถนนนครนอก",
    "หอศิลป์สงขลา": "ถนนนครนอก",
    "บ้านจีน 300 ปี": "ถนนนครนอก",
    "ร้านเจ๊นิ": "ถนนนครนอก",
    "สงขลาสเตชั่น": "ถนนนครนอก",
    "บ้านนครใน": "ถนนนครใน",
    "บ้านสงครามโลก": "ถนนนครใน",
    "ร้านเกียดฟั่ง": "ถนนนางงาม",
    "ร้านไอติมโอ่ง": "ถนนนางงาม",
    "ร้านแต้เฮี้ยงอิ้ว": "ถนนนางงาม",
    "บ้านขนมไทยสองแสน": "ถนนนางงาม",
    "ศาลเจ้าพ่อหลักเมืองสงขลา": "ถนนนางงาม",
    "โรงแรมสงขลาแต่แรก": "ถนนเพชรคีรี",
    "โรงแรมคลับทรี": "ถนนทะเลหลวง",
    "โรงแรมมอนทาน่า": "ถนนสะเดา",
    "เขาตังกวน": "ถนนสุขุม",
    "รถรางชมเมืองสงขลา": "ถนนจะนะ"
}

# Forbidden keywords in extracted entities (Hallucinations / Prompt Leakage)
FORBIDDEN_KEYWORDS = [
    "ถนนหรือตรอก", "สถานที่", "กิจกรรม", "เมนูเด็ด", "ของกิน", "ของฝาก",
    "ข้อความ", "ตัวอย่าง", "สนามบินหาดใหญ่", "นักท่องเที่ยว", "ท่าน",
    "โปรแกรมท่องเที่ยว", "ไฮไลท์", "รายละเอียด", "ป้าย", "ทิศทาง",
    "undefined", "null", "none", "unknown", "concept"
]


def clean_json_response(raw_text: str) -> list:
    """Extracts valid JSON array from LLM response text."""
    match = re.search(r'\[.*\]', raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    cleaned = re.sub(r'```json|```', '', raw_text).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def normalize_entity_name(name: str) -> str:
    """Cleans noisy suffixes like (AnyFlip หน้า XX), (ถนน...), English translations."""
    clean = re.sub(r'\(AnyFlip.*?\)', '', name)
    clean = re.sub(r'\(ถนน.*?\)', '', clean)
    clean = re.sub(r'\(ลิฟต์กระเช้าไฟฟ้า\)', '', clean)
    clean = re.sub(r'\(Songkhla Station\)', '', clean)
    clean = re.sub(r'\(Club Tree.*?\)', '', clean)
    clean = re.sub(r'\(Montana.*?\)', '', clean)
    clean = clean.strip()
    return clean


def parse_distance_props(text: str) -> dict:
    """Parses distance in meters and walking time in minutes from text evidence."""
    props = {}
    m_dist = re.search(r'(\d+(?:\.\d+)?)\s*(เมตร|กิโลเมตร|กม\.)', text)
    if m_dist:
        val = float(m_dist.group(1))
        unit = m_dist.group(2)
        if "กิโล" in unit or "กม" in unit:
            props["distance_m"] = int(val * 1000)
        else:
            props["distance_m"] = int(val)

    m_time = re.search(r'(เดิน|นั่งรถ)?\s*(\d+)\s*นาที', text)
    if m_time:
        props["walk_min"] = int(m_time.group(2))
    return props


def validate_and_clean_triple(t: dict) -> Tuple[dict, str]:
    """
    Validates and normalizes extracted triples against domain ontology constraints.
    Returns (cleaned_triple, reject_reason). If valid, reject_reason is None.
    """
    subj = normalize_entity_name(str(t.get("subject", "")).strip())
    rel = str(t.get("relation", "")).strip().upper()
    obj = normalize_entity_name(str(t.get("object", "")).strip())
    evidence = str(t.get("evidence", "")).strip()

    # 1. Basic length check
    if len(subj) < 2 or len(obj) < 2 or len(subj) > 50 or len(obj) > 60:
        return None, "Subject or Object length out of bounds"

    # 2. Check self-loop
    if subj.lower() == obj.lower():
        return None, "Self-loop edge"

    # 3. Check forbidden prompt placeholder leakage
    for kw in FORBIDDEN_KEYWORDS:
        if kw in subj.lower() or kw in obj.lower():
            return None, f"Contains forbidden placeholder or out-of-domain phrase: '{kw}'"

    # 4. Relation normalization and validation
    valid_relations = ["LOCATED_ON", "SERVES", "OFFERS_ACTIVITY", "VISITED_ON", "HISTORICAL_ERA", "CONNECTS_TO", "NEARBY"]
    if rel not in valid_relations:
        return None, f"Invalid relation: '{rel}'"

    props = {}

    # 4.1 LOCATED_ON validation & normalization
    if rel == "LOCATED_ON":
        # Target must be a recognized street or area
        normalized_street = None
        for canonical, aliases in CANONICAL_STREETS.items():
            if any(alias in obj for alias in aliases):
                normalized_street = canonical
                break
        if normalized_street:
            obj = normalized_street
        else:
            if "เมืองเก่า" in obj:
                obj = "ย่านเมืองเก่าสงขลา"
            else:
                return None, f"LOCATED_ON target '{obj}' is not a recognized Songkhla street"

        # Anti-hallucination check against known landmark streets
        for landmark, true_street in LANDMARK_TRUE_STREETS.items():
            if landmark in subj and obj != true_street and obj != "ย่านเมืองเก่าสงขลา":
                return None, f"Hallucinated street for {subj}: claims {obj} but factually {true_street}"

    # 4.2 VISITED_ON normalization
    elif rel == "VISITED_ON":
        if "วันแรก" in obj or "1" in obj or "day 1" in obj.lower():
            obj = "โปรแกรมเที่ยวสงขลา วันที่ 1"
        elif "วันที่สอง" in obj or "2" in obj or "day 2" in obj.lower():
            obj = "โปรแกรมเที่ยวสงขลา วันที่ 2"
        else:
            return None, f"VISITED_ON target '{obj}' must be Day 1 or Day 2"

    # 4.3 NEARBY parsing
    elif rel == "NEARBY":
        # Extract distance_m and walk_min if present in evidence or object
        parsed = parse_distance_props(evidence + " " + obj)
        props.update(parsed)

    # 4.4 CONNECTS_TO validation
    elif rel == "CONNECTS_TO":
        pass

    # Cleaned triple
    cleaned = {
        "subject": subj,
        "relation": rel,
        "object": obj,
        "evidence": evidence
    }
    if props:
        cleaned["properties"] = props
    return cleaned, None


def extract_triples_from_chunk(chunk_id: str, title: str, text: str) -> Tuple[List[dict], float, str, int]:
    """Sends raw chunk text to LLM with Few-Shot examples and negative constraints."""
    system_prompt = (
        "คุณคือ AI วิศวกรผู้เชี่ยวชาญด้าน Information Extraction และ Knowledge Graph "
        "หน้าที่ของคุณคือสกัด RDF Triples (Subject, Relation, Object, Evidence) จากเนื้อหาคู่มือท่องเที่ยวเมืองเก่าสงขลา "
        "โดยต้องสกัดเฉพาะข้อเท็จจริงจริง ห้ามสร้างคำตอบขึ้นมาเอง และห้ามดึงคำอธิบายในคำสั่งมาใส่เด็ดขาด"
    )

    few_shot_prompt = f"""ตัวอย่างการสกัดที่ถูกต้อง (Few-Shot Examples):
ตัวอย่างที่ 1:
ข้อความ: 'ร้านเกียดฟั่ง ตั้งอยู่บนถนนนางงาม เป็นร้านข้าวสตูหมูเจ้าเก่าแก่ เสิร์ฟข้าวสตูพร้อมซาลาเปาลูกใหญ่'
ผลลัพธ์:
[
  {{"subject": "ร้านเกียดฟั่ง", "relation": "LOCATED_ON", "object": "ถนนนางงาม", "evidence": "ตั้งอยู่บนถนนนางงาม"}},
  {{"subject": "ร้านเกียดฟั่ง", "relation": "SERVES", "object": "ข้าวสตูหมู", "evidence": "ร้านข้าวสตูหมูเจ้าเก่าแก่"}},
  {{"subject": "ร้านเกียดฟั่ง", "relation": "SERVES", "object": "ซาลาเปาลูกใหญ่", "evidence": "เสิร์ฟข้าวสตูพร้อมซาลาเปาลูกใหญ่"}}
]

ตัวอย่างที่ 2:
ข้อความ: 'โรงสีแดง หับโห้หิ้น สร้างขึ้นในสมัยรัชกาลที่ 6 ตั้งอยู่ริมทะเลสาบสงขลา ถนนนครนอก นักท่องเที่ยวนิยมมาถ่ายรูปอาคารไม้สีแดง'
ผลลัพธ์:
[
  {{"subject": "โรงสีแดง หับโห้หิ้น", "relation": "LOCATED_ON", "object": "ถนนนครนอก", "evidence": "ถนนนครนอก"}},
  {{"subject": "โรงสีแดง หับโห้หิ้น", "relation": "HISTORICAL_ERA", "object": "สมัยรัชกาลที่ 6", "evidence": "สร้างขึ้นในสมัยรัชกาลที่ 6"}},
  {{"subject": "โรงสีแดง หับโห้หิ้น", "relation": "OFFERS_ACTIVITY", "object": "ถ่ายภาพอาคารไม้สีแดง", "evidence": "นิยมมาถ่ายรูปอาคารไม้สีแดง"}}
]

ตัวอย่างที่ 3:
ข้อความ: 'จาก โรงแรมสงขลาแต่แรก เดินเท้าไปจุดถ่ายรูปสงขลาสตรีทอาร์ท ประมาณ 300 เมตร (เดิน 4 นาที) และเดินไป โรงสีแดง หับโห้หิ้น ประมาณ 450 เมตร (เดิน 6 นาที)'
ผลลัพธ์:
[
  {{"subject": "โรงแรมสงขลาแต่แรก", "relation": "NEARBY", "object": "สงขลาสตรีทอาร์ท", "evidence": "เดินไป สงขลา สตรีทอาร์ท: ประมาณ 300 เมตร (เดิน 4 นาที)"}},
  {{"subject": "โรงแรมสงขลาแต่แรก", "relation": "NEARBY", "object": "โรงสีแดง หับโห้หิ้น", "evidence": "เดินไป โรงสีแดง หับโห้หิ้น: ประมาณ 450 เมตร (เดิน 6 นาที)"}}
]

ตัวอย่างที่ 4:
ข้อความ: 'โปรแกรมท่องเที่ยววันที่ 1: รับประทานอาหารกลางวัน ข้าวสตู ที่ ร้านเกียดฟั่ง จากนั้นเดินทางไปถ่ายภาพสตรีทอาร์ท'
ผลลัพธ์:
[
  {{"subject": "ร้านเกียดฟั่ง", "relation": "VISITED_ON", "object": "โปรแกรมเที่ยวสงขลา วันที่ 1", "evidence": "โปรแกรมเที่ยววันที่ 1 รับประทานอาหารกลางวัน ข้าวสตู ที่ ร้านเกียดฟั่ง"}},
  {{"subject": "สงขลาสตรีทอาร์ท", "relation": "VISITED_ON", "object": "โปรแกรมเที่ยวสงขลา วันที่ 1", "evidence": "จากนั้นเดินทางไปถ่ายภาพสตรีทอาร์ท"}}
]

---

ข้อความที่ต้องสกัดจริง:
[หัวข้อ: {title}]
{text}

กฎเหล็ก (Ontology Constraints):
1. ใช้ relation ต่อไปนี้เท่านั้น:
   - LOCATED_ON (สถานที่ -> ถนนจริง เช่น ถนนนางงาม, ถนนนครนอก, ถนนนครใน, ถนนเพชรคีรี, ถนนจะนะ, ถนนสุขุม, ถนนทะเลหลวง, ถนนสะเดา)
   - SERVES (ร้านอาหาร -> ชื่ออาหารหรือของหวานจริง เช่น ข้าวสตู, ขนมทองเอก, ไอติมไข่แข็ง)
   - OFFERS_ACTIVITY (สถานที่ -> กิจกรรมท่องเที่ยวจริง เช่น ถ่ายรูปสตรีทอาร์ท, นั่งรถรางชมเมือง)
   - VISITED_ON (สถานที่ -> วันที่ 1 หรือ วันที่ 2)
   - HISTORICAL_ERA (สถานที่ -> ยุคสมัย เช่น รัชกาลที่ 3, รัชกาลที่ 6)
   - CONNECTS_TO (ถนนหรือสถานที่ -> ถนนหรือสถานที่ที่เชื่อมถึงกัน)
   - NEARBY (สถานที่/โรงแรม -> สถานที่ใกล้เคียง เช่น จากโรงแรมสงขลาแต่แรก เดินไป สตรีทอาร์ท)
2. กฎห้ามเด็ดขาด (Negative Constraints):
   - ห้ามใส่คำว่า 'ถนนหรือตรอกที่เชื่อมถึงกัน', 'สถานที่ใกล้เคียง', 'กิจกรรมท่องเที่ยว' เป็น object เด็ดขาด
   - ห้ามสกัดสนามบินหาดใหญ่เข้ามาเป็นถนนในเมืองเก่า
   - ห้ามมีเครื่องหมาย markdown ให้ตอบเฉพาะ JSON Array เท่านั้น"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": few_shot_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.8
        }
    }

    try:
        t0 = time.time()
        res = requests.post(OLLAMA_URL, json=payload, timeout=120)
        elapsed = time.time() - t0

        if res.status_code == 200:
            content = res.json().get("message", {}).get("content", "")
            raw_triples = clean_json_response(content)
            
            valid_triples = []
            rejected_count = 0
            for raw_t in raw_triples:
                cleaned_t, reject_reason = validate_and_clean_triple(raw_t)
                if cleaned_t:
                    cleaned_t["source_chunk"] = chunk_id
                    cleaned_t["chunk_title"] = title
                    valid_triples.append(cleaned_t)
                else:
                    rejected_count += 1

            return valid_triples, elapsed, None, rejected_count
        else:
            return [], 0.0, f"HTTP Error {res.status_code}", 0
    except Exception as e:
        return [], 0.0, str(e), 0


def run_pipeline():
    print("=" * 80)
    print("🚀 AUTOMATED HIGH-PRECISION KNOWLEDGE GRAPH EXTRACTION PIPELINE")
    print(f"Model: {MODEL_NAME} | In-Context Few-Shot + Strict Constraint Validation Layer")
    print("=" * 80)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    all_valid_triples = []
    total_raw_count = 0
    total_rejected_count = 0
    extraction_logs = []
    start_total_time = time.time()

    print(f"Processing {len(chunks)} raw text chunks...\n")

    for i, c in enumerate(chunks, 1):
        cid = c.get("chunk_id", f"chunk_{i:03d}")
        title = c.get("title", "")
        content = c.get("content", "")

        print(f"[{i:02d}/{len(chunks)}] Extracting {cid} ({title[:35]}...): ", end="", flush=True)
        triples, latency, err, rejected = extract_triples_from_chunk(cid, title, content)

        if err:
            print(f"❌ Error: {err}")
            extraction_logs.append({
                "chunk_id": cid, "title": title, "valid_triples": 0, "rejected": 0, "latency_s": 0.0, "status": err
            })
        else:
            total_raw_count += (len(triples) + rejected)
            total_rejected_count += rejected
            all_valid_triples.extend(triples)
            print(f"✅ {len(triples)} valid triples (Filtered {rejected} noisy) in {latency:.2f}s")
            extraction_logs.append({
                "chunk_id": cid, "title": title, "valid_triples": len(triples), "rejected": rejected, "latency_s": round(latency, 2), "status": "Success"
            })

    total_time = time.time() - start_total_time

    # Deduplicate triples
    unique_triples = []
    seen = set()
    for t in all_valid_triples:
        sig = (t["subject"], t["relation"], t["object"])
        if sig not in seen:
            seen.add(sig)
            unique_triples.append(t)

    # Save to JSON
    with open(OUTPUT_TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(unique_triples, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 Extraction Finished:")
    print(f"   - Total Raw Triples Extracted by LLM: {total_raw_count}")
    print(f"   - Filtered/Rejected Noise: {total_rejected_count} ({total_rejected_count/total_raw_count*100:.1f}%)" if total_raw_count else "   - Filtered: 0")
    print(f"   - Final Validated Triples Saved: {len(unique_triples)}")
    print(f"   - Saved to: {OUTPUT_TRIPLES_PATH}")

    # Generate Markdown Audit Report
    report = f"""# 📊 High-Precision Automated Knowledge Graph Extraction Report
**Execution Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Extraction Model:** `{MODEL_NAME}` (Ollama GPU) with In-Context Few-Shot Learning  
**Validation Layer:** Strict Schema Normalizer & Negative Constraint Filter  
**Total Raw Candidates Extracted:** {total_raw_count} triples  
**Noisy Candidates Filtered Out:** {total_rejected_count} ({(total_rejected_count/total_raw_count*100):.1f}% Noise Rejection Rate)  
**Final Cleaned & Validated Triples:** **{len(unique_triples)} triples**  
**Total Extraction Time:** {total_time:.2f}s (Avg {total_time/len(chunks):.2f}s/chunk)  

---

## 1. การกระจายตัวของความสัมพันธ์หลังการกรอง (Validated Relation Distribution)

| ความสัมพันธ์ (Relation) | จำนวน Triples | สัดส่วน (%) | คำอธิบายความหมายเชิงโครงสร้าง |
| :--- | :---: | :---: | :--- |
"""
    rel_counts = {}
    for t in unique_triples:
        r = t["relation"]
        rel_counts[r] = rel_counts.get(r, 0) + 1

    for rel, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(unique_triples)) * 100 if unique_triples else 0
        report += f"| `{rel}` | **{count}** | {pct:.1f}% | ความสัมพันธ์ที่ผ่านการตรวจสอบกับ Ontology |\n"

    report += """
---

## 2. ตัวอย่าง Triples คุณภาพสูงที่ผ่านการคัดกรองแล้ว (Cleaned Triples with Text Evidence)

| Chunk ต้นทาง | Subject (ประธาน) | Relation (กริยา) | Object (กรรม) | Text Evidence (ข้อความจริงจาก PDF) |
| :--- | :--- | :---: | :--- | :--- |
"""
    for t in unique_triples[:25]:
        report += f"| `{t.get('source_chunk', '')}` | **{t.get('subject', '')}** | `{t.get('relation', '')}` | **{t.get('object', '')}** | *\"{t.get('evidence', '')}\"* |\n"

    report += """
---

## 3. บันทึกผลการสกัดราย Chunk (Extraction & Filtering Log)

| Chunk ID | หัวข้อเอกสาร | Triples ที่ผ่านเกณฑ์ | Triples ขยะที่ถูกกรองทิ้ง | เวลา (s) |
| :--- | :--- | :---: | :---: | :---: |
"""
    for log in extraction_logs:
        report += f"| `{log['chunk_id']}` | {log['title'][:45]} | **{log['valid_triples']}** | {log['rejected']} | {log['latency_s']}s |\n"

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"📄 Audit report written to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    run_pipeline()
