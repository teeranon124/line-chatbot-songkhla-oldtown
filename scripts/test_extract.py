# -*- coding: utf-8 -*-
import requests
import json

test_chunk = """
โปรแกรมเที่ยวสงขลา วันที่ 1
09:00 น. เริ่มต้นที่ หอศิลป์สงขลา (Songkhla Art Center) บนถนนนครนอก ชมนิทรรศการศิลปะ
11:00 น. เดินไปถ่ายรูปที่ โรงสีแดง หับโห้หิ้น อาคารไม้สีแดงสดริมทะเลสาบสงขลา
12:00 น. รับประทานอาหารกลางวันที่ ร้านเกียดฟั่ง บนถนนนางงาม ชิมข้าวสตูหมูและซาลาเปาลูกใหญ่
17:30 น. แวะกินของหวานที่ ร้านไอติมโอ่ง สั่งไอติมโบราณใส่ไข่แข็ง
"""

prompt = f"""จงทำ Information Extraction สกัดความสัมพันธ์ (Knowledge Graph Triples) จากข้อความต่อไปนี้:
{test_chunk}

กฎ:
1. สกัดเฉพาะข้อมูลที่ปรากฏในข้อความเท่านั้น ห้ามจินตนาการเพิ่ม
2. ใช้ Relation ต่อไปนี้เท่านั้น:
   - LOCATED_ON (สถานที่ -> ถนน)
   - VISITED_ON (สถานที่ -> เวลาหรือวันที่)
   - SERVES (ร้านอาหาร -> เมนู)
   - OFFERS_ACTIVITY (สถานที่ -> กิจกรรม)
3. ตอบกลับเป็น JSON Array ของ Triples เท่านั้น:
[
  {{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "..."}}
]"""

payload = {
    "model": "qwen2.5:3b",
    "messages": [
        {"role": "system", "content": "You are a professional Information Extraction and Knowledge Graph Engineer. Output strictly valid JSON."},
        {"role": "user", "content": prompt}
    ],
    "stream": False,
    "options": {"temperature": 0.1}
}

res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
print("Status:", res.status_code)
print("Result:")
print(res.json()["message"]["content"])
