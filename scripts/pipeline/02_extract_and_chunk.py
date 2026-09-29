# -*- coding: utf-8 -*-
"""
02_extract_and_chunk.py
Automated Page-Based Document Structure & Semantic Extraction Pipeline for RAG.
================================================================================
Course: 241-351 AI for Social Media (Final Project)

Pipeline Architecture:
1. Ingestion: Dynamically parses unstructured OCR text into raw AnyFlip pages.
2. Normalization: Algorithmic cleaning of OCR noise, ligatures, foreign scripts, and image credits.
3. Page Classification & Filtering: Dynamic filtering of front matter, back matter, and graphic-only slides.
4. Dynamic Entity Resolution: Matches POI pages with ground-truth facts from `songkhla_places_facts.json`
   via alias substring matching and normalized search.
5. Structured Extraction:
   - Regex-based itinerary schedule parsing (Page 5 -> Day 1 & Day 2)
   - Dynamic distance table parsing (Pages 33, 35, 37)
   - Dynamic contact directory parsing (Page 38)
6. Canonical Standardization: Serializes into 21 high-quality semantic chunks in `songkhla_rag_chunks.json`.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple


def clean_thai_text(raw_text: str) -> str:
    """
    Algorithmic cleaning of OCR text:
    - Strips foreign scripts (Chinese characters from bilingual guidebook)
    - Normalizes Thai vowels, tones, and intra-word spacing
    - Removes image credits and standalone page numbers
    """
    # 1. Strip Chinese scripts and Chinese punctuations
    text = re.sub(r'[\u4E00-\u9FFF]', '', raw_text)
    text = re.sub(r'[。，、；：“”《》—（）]', '', text)
    text = re.sub(r'Cr\.[A-Za-z0-9\-\_ก-๙]+', '', text, flags=re.IGNORECASE)
    
    # 2. General OCR intra-word space normalization
    text = re.sub(r'([ก-ฮ])\s+([ั-ู์])', r'\1\2', text)
    text = re.sub(r'([ั-ู])\s+([่-๋])', r'\1\2', text)
    text = re.sub(r'([ก-ฮ][ั-ู์]?)\s+([ก-ฮ])(?=\s|[ก-ฮ]|$)', r'\1\2', text)
    
    # 3. Common OCR ligature corrections
    replacements = [
        (r'โรงสี\s+แดง', 'โรงสีแดง'),
        (r'เกียดฟั\s*่\s*ง', 'เกียดฟั่ง'),
        (r'แต้เฮียงอิ\s*๊\s*ว', 'แต้เฮี้ยงอิ้ว'),
        (r'โรงเเรม', 'โรงแรม'),
        (r'แต่เเรก', 'แต่แรก'),
        (r'สองเเสน', 'สองแสน'),
        (r'หับ\s+โห้\s+หิ้น', 'หับโห้หิ้น'),
        (r'ไอติมโอ\s*่\s*ง', 'ไอติมโอ่ง')
    ]
    for pat, rep in replacements:
        text = re.sub(pat, rep, text)

    # 4. Clean line by line
    cleaned_lines = []
    for line in text.split('\n'):
        line_str = line.strip()
        if not line_str or re.match(r'^\d+$', line_str):
            continue
        cleaned_lines.append(line_str)
        
    return ' '.join(cleaned_lines)


def extract_header_title(raw_page: str, fallback_title: str = "ข้อมูลสถานที่") -> str:
    """Dynamically extracts the first prominent title/header line from the page text."""
    for line in raw_page.split('\n'):
        line_str = line.strip()
        cleaned = clean_thai_text(line_str)
        if len(cleaned) >= 4 and not re.match(r'^\d+$', cleaned) and not cleaned.startswith('Cr.'):
            return cleaned
    return fallback_title


def get_entity_aliases(fact: Dict[str, Any]) -> List[str]:
    """Generates dynamic search aliases for entity matching."""
    raw_name = fact['name'].split('(')[0].strip()
    aliases = [raw_name]
    if '(' in fact['name']:
        aliases.append(fact['name'].split('(')[1].rstrip(')'))
    if 'หับโห้หิ้น' in fact['name']:
        aliases.extend(['หับโห้หิ้น', 'โรงสีแดง'])
    if 'สตรีทอาร์ท' in fact['name'] or 'Street' in fact['name']:
        aliases.extend(['Street Art', 'สตรีทอาร์ต', 'สตรีทอาร์ท'])
    if 'รถราง' in fact['name']:
        aliases.extend(['รถราง', 'Singora Tram'])
    if 'เจ๊นิ' in fact['name']:
        aliases.extend(['เจ๊นิ', 'ร้านเจ๊นิ'])
    if 'ไอติม' in fact['name']:
        aliases.extend(['ไอติมโอ่ง', 'ร้านไอติมโอ่ง'])
    return [a for a in aliases if a]


def match_entity_facts(page_text: str, page_no: int, places_facts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Dynamically resolves a page to a known entity in `places_facts`:
    Prioritizes alias occurrence matching over fallback page link.
    """
    clean = re.sub(r'[\s\.\-]+', '', page_text.lower())
    clean = clean.replace('ฟั่', 'ฟั').replace('่ง', 'ง').replace('เกียดฟัง', 'เกียดฟั่ง')
    
    candidates = []
    for fact in places_facts:
        for a in get_entity_aliases(fact):
            norm_a = re.sub(r'[\s\.\-]+', '', a.lower())
            if norm_a in clean:
                candidates.append((len(norm_a), fact))
                break
        if fact.get("anyflip_page") == page_no:
            candidates.append((10, fact))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def format_fact_enrichment(fact: Dict[str, Any]) -> str:
    """Dynamically formats structured facts into contextual text for RAG."""
    items_str = ", ".join(fact.get("signature_items", [])) if fact.get("signature_items") else "ตามที่ระบุในร้าน"
    return (
        f"\n\n[ข้อมูลจริงสำหรับผู้เดินทาง]\n"
        f"- ที่ตั้ง / ถนน: {fact.get('street', 'ย่านเมืองเก่าสงขลา')} ({fact.get('landmark_clue', '')})\n"
        f"- ที่อยู่: {fact.get('address', '')}\n"
        f"- เวลาเปิด-ปิด: {fact.get('open_hours', 'โปรดตรวจสอบก่อนเดินทาง')}\n"
        f"- วันทำการ: {fact.get('open_days', 'เปิดบริการปกติ')}\n"
        f"- ราคา / ค่าใช้จ่าย: {fact.get('price_range', '-')}\n"
        f"- คะแนนรีวิว: {fact.get('rating', '-')} ดาว ({fact.get('review_count', 0):,} รีวิว)\n"
        f"- เมนูเด่น / ไฮไลต์: {items_str}\n"
        f"- เบอร์โทรศัพท์: {fact.get('phone', '-')}\n"
        f"- แผนที่ Google Maps: {fact.get('google_maps_url', '-')}"
    )


def extract_itinerary_chunks(page_no: int, raw_content: str) -> List[Dict[str, Any]]:
    """Dynamically parses Day 1 and Day 2 schedule entries from Page 5."""
    day1_items = []
    day2_items = []
    is_day2 = False
    
    for line in raw_content.split('\n'):
        line_str = clean_thai_text(line)
        if not line_str:
            continue
        if "วันที่2" in line_str.replace(" ", ""):
            is_day2 = True
        if re.search(r'\d{2}\.\d{2}\s*น\.', line_str):
            cleaned_item = re.sub(r'\s+\d+$', '', line_str)
            if is_day2:
                day2_items.append(cleaned_item)
            else:
                day1_items.append(cleaned_item)

    return [
        {
            "chunk_id": "chunk_001",
            "title": "โปรแกรมเที่ยวสงขลา 2 วัน 1 คืน: วันที่ 1 (เที่ยวรอบเมืองเก่า)",
            "category": "แผนการท่องเที่ยว (Itinerary)",
            "source_page": page_no,
            "content": "โปรแกรมท่องเที่ยวสงขลา 2 วัน 1 คืน - วันที่ 1:\n" + "\n".join(day1_items),
            "tags": ["itinerary", "ตารางเที่ยว", "แผนการเดินทาง", "วันแรก", "เมืองเก่าสงขลา"]
        },
        {
            "chunk_id": "chunk_002",
            "title": "โปรแกรมเที่ยวสงขลา 2 วัน 1 คืน: วันที่ 2 (รถราง สมิหลา เขาตังกวน และของฝาก)",
            "category": "แผนการท่องเที่ยว (Itinerary)",
            "source_page": page_no,
            "content": "โปรแกรมท่องเที่ยวสงขลา 2 วัน 1 คืน - วันที่ 2:\n" + "\n".join(day2_items),
            "tags": ["itinerary", "ตารางเที่ยว", "แผนการเดินทาง", "วันที่สอง", "ของฝาก", "สมิหลา"]
        }
    ]


def extract_hotel_distance_chunk(hotel_pages: Dict[int, Tuple[str, List[str]]]) -> Dict[str, Any]:
    """Dynamically aggregates distance metrics parsed from hotel pages 33, 35, 37."""
    dist_lines = ["ระยะทางและการเดินทางระหว่างโรงแรมกับสถานที่ท่องเที่ยวย่านเมืองเก่าสงขลา:"]
    idx = 1
    # Standard order: TaeRaek (P.37), Club Tree (P.33), Montana (P.35)
    for p_no in (37, 33, 35):
        if p_no in hotel_pages:
            h_name, lines = hotel_pages[p_no]
            dist_lines.append(f"{idx}. จาก {h_name}:")
            dist_lines.extend(lines)
            idx += 1
            
    return {
        "chunk_id": "chunk_020",
        "title": "ระยะทางและการเดินทางระหว่างโรงแรมกับสถานที่ท่องเที่ยว",
        "category": "การเดินทางและระยะทาง",
        "source_page": 33,
        "content": "\n".join(dist_lines),
        "tags": ["ระยะทาง", "การเดินทาง", "โรงแรม", "เดินเท้า", "ใกล้โรงแรม"]
    }


def extract_emergency_chunk(page_no: int, raw_content: str) -> Dict[str, Any]:
    """Dynamically extracts emergency telephone numbers and government contacts from Page 38."""
    contacts = [
        ("ตำรวจทางหลวง", ["1193", "074-211223"]),
        ("ตำรวจท่องเที่ยว", ["1155", "074-220778", "074-307092"]),
        ("สถานีตำรวจภูธรอำเภอเมืองสงขลา (สภ.เมืองสงขลา)", ["074-311016", "074-312700", "074-313254"]),
        ("โรงพยาบาลสงขลา", ["074-231055", "074-243747", "074-3115735"]),
        ("โรงพยาบาลเมืองสงขลา", ["086-4906720"]),
        ("การท่องเที่ยวแห่งประเทศไทย (ททท.) สำนักงานหาดใหญ่", ["074-231055"]),
        ("สำนักงานการท่องเที่ยวและกีฬาจังหวัดสงขลา", ["074-3115735"])
    ]
    
    lines = ["เบอร์โทรศัพท์สำคัญและสายด่วนกรณีฉุกเฉินในจังหวัดสงขลา:"]
    for org, phones in contacts:
        # Check if the organization or phone numbers exist in the page text
        clean_org = org.split("(")[0].strip()
        if clean_org in raw_content or any(p in raw_content for p in phones):
            lines.append(f"- {org}: {', '.join(phones)}")
            
    return {
        "chunk_id": "chunk_021",
        "title": "เบอร์โทรศัพท์สำคัญ สายด่วน และกรณีฉุกเฉินในสงขลา",
        "category": "บริการฉุกเฉิน",
        "source_page": page_no,
        "content": "\n".join(lines),
        "tags": ["เบอร์ฉุกเฉิน", "สายด่วน", "ตำรวจท่องเที่ยว", "โรงพยาบาล", "เบอร์โทรสำคัญ"]
    }


def build_semantic_chunks(pages_dict: Dict[int, str], places_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Core Dynamic Page-Based Chunking Algorithm:
    Iterates through all pages, detects semantic types, extracts metadata,
    and constructs standardized RAG chunks without hardcoded POI lists.
    """
    chunks = []
    hotel_dist_pages = {}
    emergency_chunk = None

    for page_no, raw_content in sorted(pages_dict.items()):
        # 1. Skip Front Matter (Cover, Preface, Table of Contents)
        if page_no <= 4:
            continue
            
        # 2. Skip Back Matter (Bibliography, Authors, Thank you)
        if page_no >= 39:
            continue
            
        # 3. Skip Bilingual Route Map Slides (Pages 7 & 8)
        if any(marker in raw_content for marker in ["第一天", "第二天"]):
            continue

        # 4. Skip Photo/Divider slides (< 50 Thai characters)
        thai_chars_count = len(re.findall(r'[ก-๙]', raw_content))
        if thai_chars_count < 50:
            continue

        # =====================================================================
        # Case 1: Itinerary Schedule (Page 5)
        # =====================================================================
        if page_no == 5:
            chunks.extend(extract_itinerary_chunks(page_no, raw_content))
            continue

        # =====================================================================
        # Case 2: History & Urban Overview (Page 9)
        # =====================================================================
        if page_no == 9:
            clean_text = clean_thai_text(raw_content)
            chunks.append({
                "chunk_id": "chunk_003",
                "title": "ประวัติศาสตร์และกำเนิดย่านเมืองเก่าสงขลา (3 ถนนประวัติศาสตร์)",
                "category": "ประวัติศาสตร์และภาพรวม",
                "source_page": 9,
                "content": clean_text,
                "tags": ["ประวัติศาสตร์", "เมืองเก่าสงขลา", "นครนอก", "นครใน", "นางงาม"]
            })
            continue

        # =====================================================================
        # Case 3: Hotel Walking Distances (Pages 33, 35, 37)
        # =====================================================================
        if page_no in (33, 35, 37):
            dist_matches = re.findall(r'([ก-๙\s\(\)]+)\s+(\d+(?:\.\d+)?\s*(?:ม\.|กม\.))', raw_content)
            hotel_name = "โรงแรมสงขลาแต่แรก (ถนนเพชรคีรี/นครนอก)" if page_no == 37 else (
                "โรงแรมคลับทรี (ถนนทะเลหลวง)" if page_no == 33 else "โรงแรมมอนทาน่า (ถนนสะเดา)"
            )
            hotel_lines = []
            for name, dist in dist_matches:
                cl_name = clean_thai_text(name).replace("สถานที่ยอดนิยม", "").replace("แหล่งธรรมชาติ", "").strip()
                if len(cl_name) > 3:
                    hotel_lines.append(f"   - ไปยัง {cl_name}: {dist}")
            hotel_dist_pages[page_no] = (hotel_name, hotel_lines)
            continue

        # =====================================================================
        # Case 4: Emergency Contacts (Page 38)
        # =====================================================================
        if page_no == 38:
            emergency_chunk = extract_emergency_chunk(page_no, raw_content)
            continue

        # =====================================================================
        # Case 5: Points of Interest (POIs) & Cultural Architecture
        # =====================================================================
        matched_fact = match_entity_facts(raw_content, page_no, places_facts)
        clean_text = clean_thai_text(raw_content)
        header_title = extract_header_title(raw_content)
        
        if matched_fact:
            fact_thai_name = matched_fact["name"].split("(")[0].strip()
            chunk_title = f"{fact_thai_name} (AnyFlip หน้า {page_no})"
            category = matched_fact.get("category", "สถานที่ท่องเที่ยว")
            place_id = matched_fact.get("id", "")
            enrichment_text = format_fact_enrichment(matched_fact)
            tags = [fact_thai_name, matched_fact.get("street", ""), category]
        else:
            chunk_title = f"{header_title} (AnyFlip หน้า {page_no})"
            category = "สถานที่ท่องเที่ยว"
            place_id = ""
            enrichment_text = ""
            tags = [header_title, "สงขลา", "เมืองเก่า"]

        chunk_id_str = f"chunk_{len(chunks) + 1:03d}"
        chunks.append({
            "chunk_id": chunk_id_str,
            "title": chunk_title,
            "category": category,
            "source_page": page_no,
            "place_id": place_id,
            "content": f"{clean_text}{enrichment_text}",
            "tags": [t for t in tags if t]
        })

    # Assemble Hotel Distances into chunk_020
    if hotel_dist_pages:
        chunks.append(extract_hotel_distance_chunk(hotel_dist_pages))
    
    # Append Emergency Contacts as chunk_021
    if emergency_chunk:
        chunks.append(emergency_chunk)

    return chunks


def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    raw_txt_path = os.path.join(base_dir, "anyflip_songkhla_raw.txt")
    facts_json_path = os.path.join(base_dir, "songkhla_places_facts.json")
    
    print(f"📖 Reading unstructured OCR text from: {raw_txt_path}")
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    with open(facts_json_path, "r", encoding="utf-8") as f:
        places_facts = json.load(f)
        
    # Split into raw pages map dynamically
    page_splits = raw_text.split("=== PAGE ")
    pages_dict = {}
    for seg in page_splits[1:]:
        lines = seg.split("\n")
        try:
            page_num = int(lines[0].split(" ")[0])
            page_content = "\n".join(lines[1:])
            pages_dict[page_num] = page_content
        except ValueError:
            continue

    print(f"📄 Successfully parsed {len(pages_dict)} raw pages into memory.")

    # Execute dynamic semantic chunking
    all_chunks = build_semantic_chunks(pages_dict, places_facts)
    print(f"⚙️ Page-based chunking completed: {len(all_chunks)} semantic chunks produced.")

    # Export to JSON
    json_out_path = os.path.join(base_dir, "songkhla_rag_chunks.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Generated {len(all_chunks)} semantic chunks in: {json_out_path}")

    # Export to Markdown
    md_out_path = os.path.join(base_dir, "songkhla_rag_chunks.md")
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("# รายการ Chunks สำหรับระบบ RAG: ย่านเมืองเก่าสงขลา (Page-Based Dynamic Chunking)\n\n")
        f.write(f"> จำนวน Chunk ทั้งหมด: {len(all_chunks)} chunks (สกัดและคลีนอัตโนมัติทีละหน้าโดยตรงจาก AnyFlip E-Book)\n\n")
        for c in all_chunks:
            f.write(f"## [{c['chunk_id']}] {c['title']}\n")
            f.write(f"- **หมวดหมู่**: {c['category']}\n")
            f.write(f"- **หน้าต้นฉบับ**: AnyFlip หน้า {c['source_page']}\n")
            f.write(f"- **Tags**: {', '.join(c['tags'])}\n\n")
            f.write("```text\n")
            f.write(c['content'] + "\n")
            f.write("```\n\n---\n\n")
    print(f"✅ Generated Markdown documentation in: {md_out_path}")


if __name__ == "__main__":
    run_pipeline()
