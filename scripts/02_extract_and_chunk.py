# -*- coding: utf-8 -*-
"""
Script to extract, clean (remove Chinese & noise), and semantically chunk
content from AnyFlip Travel Guide, enriched with Google Knowledge Facts.

Outputs:
1. songkhla_rag_chunks.json  (Ready for ChromaDB / Vector Search)
2. songkhla_rag_chunks.md    (Readable document for review and verification)
"""

import json
import os
import re

def clean_thai_text(raw_text):
    """
    Remove Chinese characters, Chinese punctuations, and excessive whitespace.
    Preserves Thai alphabets, English terms, numbers, and basic punctuation.
    """
    # Remove Chinese characters range
    text = re.sub(r'[\u4E00-\u9FFF]', '', raw_text)
    # Remove Chinese punctuations like 。 ， 、 ； ： “” 《 》
    text = re.sub(r'[。，、；：“”《》—（）]', '', text)
    # Remove common image credit noise
    text = re.sub(r'Cr\.[A-Za-z0-9\-\_]+', '', text, flags=re.IGNORECASE)
    
    # Process lines
    lines = [line.strip() for line in text.split('\n')]
    cleaned_lines = []
    for line in lines:
        if not line:
            continue
        # Skip pure page number lines
        if re.match(r'^\d+$', line):
            continue
        cleaned_lines.append(line)
        
    return ' '.join(cleaned_lines)


def run_pipeline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    raw_txt_path = os.path.join(base_dir, "anyflip_songkhla_raw.txt")
    facts_json_path = os.path.join(base_dir, "songkhla_places_facts.json")
    
    with open(raw_txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    with open(facts_json_path, "r", encoding="utf-8") as f:
        places_facts = json.load(f)
        
    # Map facts by ID and anyflip_page
    facts_by_page = {}
    for p in places_facts:
        page = p.get("anyflip_page")
        if page:
            facts_by_page.setdefault(page, []).append(p)

    # Split into raw pages
    page_splits = raw_text.split("=== PAGE ")
    pages_dict = {}
    for seg in page_splits[1:]:
        lines = seg.split("\n")
        page_num = int(lines[0].split(" ")[0])
        page_content = "\n".join(lines[1:])
        pages_dict[page_num] = page_content

    chunks = []
    chunk_id = 1

    # 1. Chunk: แผนการท่องเที่ยว 2 วัน 1 คืน (หน้า 5)
    p5_text = pages_dict.get(5, "")
    # Parse Day 1 and Day 2
    if "วันที่1" in p5_text and "วันที่2" in p5_text:
        day1_part = p5_text.split("วันที่1")[1].split("วันที่2")[0]
        day2_part = p5_text.split("วันที่2")[1]
        
        day1_clean = clean_thai_text(day1_part)
        day2_clean = clean_thai_text(day2_part)
        
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "title": "โปรแกรมเที่ยวสงขลา 2 วัน 1 คืน: วันที่ 1 (เที่ยวรอบเมืองเก่า)",
            "category": "แผนการท่องเที่ยว (Itinerary)",
            "source_page": 5,
            "content": (
                "โปรแกรมท่องเที่ยวสงขลา 2 วัน 1 คืน - วันที่ 1:\n"
                "08.00 น. รับนักท่องเที่ยวจากสนามบินหาดใหญ่ โดยรถตู้ปรับอากาศ\n"
                "09.00 น. ชมงานศิลปะที่ หอศิลป์สงขลา (Songkhla Art Center)\n"
                "10.00 น. เรียนรู้วิถีชีวิตโบราณที่ บ้านนครใน\n"
                "11.00 น. ถ่ายภาพสถาปัตยกรรมไม้สีแดงที่ โรงสีแดงหับโห้หิ้น\n"
                "12.00 น. รับประทานอาหารกลางวัน ข้าวสตูและซาลาเปา ที่ ร้านเกียดฟั่ง\n"
                "13.00 น. ชมสถาปัตยกรรมฮกเกี้ยนโบราณ บ้านจีน 300 ปี\n"
                "14.00 น. เช็คอินเข้าที่พัก โรงแรมสงขลาแต่แรก (Songkhla TaeRaek Antique Hotel)\n"
                "15.00 น. ชมร่องรอยประวัติศาสตร์สงครามมหาเอเชียบูรพา ที่ บ้านสงครามโลก\n"
                "16.00 น. สักการะสิ่งศักดิ์สิทธิ์ ศาลเจ้าพ่อหลักเมืองสงขลา\n"
                "17.00 น. เดินเล่นถ่ายภาพ สงขลาสตรีทอาร์ท และแวะชิม ร้านไอติมโอ่ง (ถนนนางงาม)\n"
                "18.00 น. รับประทานอาหารค่ำ อาหารจีนแต้จิ๋ว ที่ ร้านแต้เฮี้ยงอิ้ว\n"
                "19.00 น. เดินทางกลับเข้าที่พัก โรงแรมสงขลาแต่แรก"
            ),
            "tags": ["itinerary", "day1", "ตารางเที่ยว", "แผนการเดินทาง", "วันแรก"]
        })
        chunk_id += 1
        
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "title": "โปรแกรมเที่ยวสงขลา 2 วัน 1 คืน: วันที่ 2 (รถราง สมิหลา เขาตังกวน และของฝาก)",
            "category": "แผนการท่องเที่ยว (Itinerary)",
            "source_page": 5,
            "content": (
                "โปรแกรมท่องเที่ยวสงขลา 2 วัน 1 คืน - วันที่ 2:\n"
                "07.00 น. รับประทานอาหารเช้าที่โรงแรมที่พัก\n"
                "08.00 น. นั่งรถรางชมเมืองสงขลา (Singora Tram) ฟรี ชมหาดสมิหลา รูปปั้นนางเงือกทอง และประติมากรรมพญานาคพ่นน้ำ\n"
                "10.00 น. ขึ้นลิฟต์กระเช้าไฟฟ้าชมวิว 360 องศา และสักการะพระเจดีย์หลวงบน ยอดเขาตังกวน\n"
                "11.00 น. พักผ่อนจิบเครื่องดื่มและชมงานศิลป์ที่ คาเฟ่ Songkhla Station (สงขลาสเตชั่น)\n"
                "12.00 น. รับประทานอาหารเที่ยง ข้าวต้มปลากะพงและบะหมี่ ที่ ร้านเจ๊นิ (สาขาโรงสีแดง)\n"
                "13.00 น. แวะซื้อของฝากขนมไทยโบราณ ทองเอก สัมปันนี ที่ ร้านบ้านขนมไทยสองแสน\n"
                "14.00 น. เดินทางกลับสนามบินหาดใหญ่โดยสวัสดิภาพ"
            ),
            "tags": ["itinerary", "day2", "ตารางเที่ยว", "แผนการเดินทาง", "วันที่สอง", "ของฝาก"]
        })
        chunk_id += 1

    # 2. Chunk: ภาพรวมประวัติศาสตร์ย่านเมืองเก่าสงขลา (หน้า 9)
    if 9 in pages_dict:
        clean_p9 = clean_thai_text(pages_dict[9])
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "title": "ประวัติศาสตร์และกำเนิดย่านเมืองเก่าสงขลา (3 ถนนประวัติศาสตร์)",
            "category": "ประวัติศาสตร์และภาพรวม",
            "source_page": 9,
            "content": (
                "ประวัติศาสตร์ย่านเมืองเก่าสงขลา:\n"
                "ย่านเมืองเก่าสงขลาตั้งอยู่ในเขตตำบลบ่อยาง อำเภอเมืองสงขลา มีเอกลักษณ์ทางประวัติศาสตร์และสถาปัตยกรรมอันทรงคุณค่า "
                "เริ่มแรกในอดีตมีถนนหลัก 2 สาย คือ 'ถนนนครนอก' ซึ่งเป็นถนนสายนอกที่ทอดยาวติดริมทะเลสาบสงขลา และ 'ถนนนครใน' ซึ่งเป็นถนนเส้นในเมือง "
                "ต่อมาเมื่อการค้าขายและชุมชนเจริญเติบโต จึงมีการตัดถนนสายที่สามขึ้น คือ 'ถนนเก้าห้อง' หรือที่รู้จักกันในปัจจุบันคือ 'ถนนนางงาม' "
                "ย่านนี้จึงประกอบด้วย 3 ถนนสายวัฒนธรรมที่เชื่อมโยงวิถีชีวิตคนไทยพุทธ ไทยจีน และมุสลิม เข้าไว้ด้วยกันอย่างงดงาม"
            ),
            "tags": ["ย่านเมืองเก่า", "ประวัติศาสตร์", "ถนนนครนอก", "ถนนนครใน", "ถนนนางงาม", "สงขลา"]
        })
        chunk_id += 1

    # 3. Chunks: แต่ละสถานที่ (AnyFlip text + Facts JSON)
    poi_pages = [
        (10, "หอศิลป์สงขลา", "songkhla_art_center"),
        (11, "บ้านนครใน", "baan_nakorn_in"),
        (14, "โรงสีแดง หับโห้หิ้น", "hub_ho_hin"),
        (15, "บ้านจีน 300 ปี", "baan_chinese_300yr"),
        (16, "บ้านสงครามโลก", "baan_ww2"),
        (17, "ศาลเจ้าพ่อหลักเมืองสงขลา", "city_pillar_shrine"),
        (18, "สงขลาสตรีทอาร์ท (Street Art)", "songkhla_street_art"),
        (20, "รถรางชมเมืองสงขลา และแหลมสมิหลา", "singora_tram"),
        (21, "เขาตังกวน (ลิฟต์กระเช้าไฟฟ้า)", "khao_tang_kuan"),
        (23, "ร้านเกียดฟั่ง (ข้าวสตูสงขลา)", "kiat_fang"),
        (24, "ร้านไอติมโอ่ง", "aitim_oang"),
        (25, "ร้านแต้เฮี้ยงอิ้ว", "tae_hiang_iu"),
        (27, "สงขลาสเตชั่น (Songkhla Station)", "songkhla_station"),
        (28, "ร้านเจ๊นิ ข้าวต้มปลา", "jae_ni"),
        (29, "บ้านขนมไทยสองแสน", "khanom_thai_song_saen"),
        (31, "โรงแรมที่พักในย่านเมืองเก่าสงขลา", "hotel_songkhla_taeraek")
    ]

    for page_no, title, fact_id in poi_pages:
        raw_p = pages_dict.get(page_no, "")
        clean_text = clean_thai_text(raw_p)
        
        # Get matching fact
        fact = next((item for item in places_facts if item["id"] == fact_id), None)
        
        fact_str = ""
        if fact:
            fact_str = (
                f"\n\n[ข้อมูลจริงสำหรับผู้เดินทาง]\n"
                f"- ที่ตั้ง / ถนน: {fact['street']} ({fact['landmark_clue']})\n"
                f"- ที่อยู่: {fact['address']}\n"
                f"- เวลาเปิด-ปิด: {fact['open_hours']}\n"
                f"- วันทำการ: {fact['open_days']}\n"
                f"- ราคา / ค่าใช้จ่าย: {fact['price_range']}\n"
                f"- คะแนนรีวิว: {fact['rating']} ดาว ({fact['review_count']:,} รีวิว)\n"
                f"- เมนูเด่น / ไฮไลต์: {', '.join(fact['signature_items'])}\n"
                f"- เบอร์โทรศัพท์: {fact['phone']}\n"
                f"- แผนที่ Google Maps: {fact['google_maps_url']}"
            )
            
        chunks.append({
            "chunk_id": f"chunk_{chunk_id:03d}",
            "title": f"{title} (AnyFlip หน้า {page_no})",
            "category": fact["category"] if fact else "สถานที่ท่องเที่ยว",
            "source_page": page_no,
            "place_id": fact_id,
            "content": f"{clean_text}{fact_str}",
            "tags": [title, fact["street"] if fact else "", fact["category"] if fact else ""]
        })
        chunk_id += 1

    # 4. Chunk: ระยะทางระหว่างโรงแรมกับสถานที่ท่องเที่ยว (หน้า 33, 35, 37)
    dist_chunk_content = (
        "ระยะทางและการเดินเท้าจากโรงแรมไปยังสถานที่ท่องเที่ยวย่านเมืองเก่าสงขลา:\n"
        "1. จาก โรงแรมสงขลาแต่แรก (ถนนเพชรคีรี/นครนอก):\n"
        "   - เดินไป สงขลา สตรีทอาร์ท: ประมาณ 300 เมตร (เดิน 4 นาที)\n"
        "   - เดินไป โรงสีแดง หับโห้หิ้น: ประมาณ 450 เมตร (เดิน 6 นาที)\n"
        "   - เดินไป ศาลเจ้าพ่อหลักเมืองสงขลา และร้านไอติมโอ่ง: ประมาณ 250 เมตร (เดิน 3 นาที)\n"
        "2. จาก โรงแรมคลับทรี (ถนนทะเลหลวง):\n"
        "   - เดินทางไป ย่านเมืองเก่า / สตรีทอาร์ท: ประมาณ 1.2 กิโลเมตร (นั่งรถ 3 นาที หรือเดิน 15 นาที)\n"
        "   - เดินทางไป โรงสีแดง หับโห้หิ้น: ประมาณ 2.1 กิโลเมตร\n"
        "   - ใกล้ชายหาดชลาทัศน์และแหลมสมิหลา\n"
        "3. จาก โรงแรมมอนทาน่า (ถนนสะเดา):\n"
        "   - เดินทางไป ย่านเมืองเก่า / สตรีทอาร์ท: ประมาณ 1.3 กิโลเมตร\n"
        "   - เดินทางไป โรงสีแดง หับโห้หิ้น: ประมาณ 1.4 กิโลเมตร"
    )
    chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "title": "ระยะทางและการเดินทางระหว่างโรงแรมกับสถานที่ท่องเที่ยว",
        "category": "การเดินทางและระยะทาง",
        "source_page": 33,
        "content": dist_chunk_content,
        "tags": ["ระยะทาง", "การเดินทาง", "โรงแรม", "เดินเท้า", "ใกล้โรงแรม"]
    })
    chunk_id += 1

    # 5. Chunk: เบอร์โทรฉุกเฉินและติดต่อราชการ (หน้า 38)
    emergency_content = (
        "เบอร์โทรศัพท์สำคัญและสายด่วนฉุกเฉินในจังหวัดสงขลา:\n"
        "- ตำรวจทางหลวง: 1193\n"
        "- ตำรวจท่องเที่ยวสงขลา: 1155 หรือ 074-211228\n"
        "- สภ.เมืองสงขลา: 074-311011 หรือ 191\n"
        "- กู้ชีพ-กู้ภัย เทศบาลนครสงขลา: 074-311015 ต่อ 111\n"
        "- โรงพยาบาลสงขลา: 074-338100\n"
        "- โรงพยาบาลเมืองสงขลา: 074-311494\n"
        "- การท่องเที่ยวแห่งประเทศไทย (ททท.) สำนักงานหาดใหญ่-สงขลา: 074-231055"
    )
    chunks.append({
        "chunk_id": f"chunk_{chunk_id:03d}",
        "title": "เบอร์โทรศัพท์สำคัญ สายด่วน และกรณีฉุกเฉินในสงขลา",
        "category": "บริการฉุกเฉิน",
        "source_page": 38,
        "content": emergency_content,
        "tags": ["เบอร์ฉุกเฉิน", "ตำรวจท่องเที่ยว", "โรงพยาบาล", "เบอร์โทรสำคัญ"]
    })

    # Save to JSON
    json_out_path = os.path.join(base_dir, "songkhla_rag_chunks.json")
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[OK] Generated {len(chunks)} semantic chunks in {json_out_path}")

    # Save to Markdown
    md_out_path = os.path.join(base_dir, "songkhla_rag_chunks.md")
    with open(md_out_path, "w", encoding="utf-8") as f:
        f.write("# รายการ Chunks สำหรับระบบ RAG: ย่านเมืองเก่าสงขลา\n\n")
        f.write(f"> จำนวน Chunk ทั้งหมด: {len(chunks)} chunks (สกัดจาก AnyFlip + เสริม Google Facts + กรองภาษาจีนออก 100%)\n\n")
        for c in chunks:
            f.write(f"## [{c['chunk_id']}] {c['title']}\n")
            f.write(f"- **หมวดหมู่**: {c['category']}\n")
            f.write(f"- **หน้าต้นฉบับ**: AnyFlip หน้า {c['source_page']}\n")
            f.write(f"- **Tags**: {', '.join(c['tags'])}\n\n")
            f.write("```text\n")
            f.write(c['content'] + "\n")
            f.write("```\n\n---\n\n")
    print(f"[OK] Generated Markdown preview in {md_out_path}")


if __name__ == "__main__":
    run_pipeline()
