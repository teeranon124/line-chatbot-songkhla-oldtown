# -*- coding: utf-8 -*-
"""
RAG Engine Orchestrator for Songkhla Old Town Assistant.
Features:
- Dual-Engine LLM Generation (Local Ollama vs Cloud Groq API)
- Intelligent Model Router (Factual -> Local, Complex Reasoning -> Cloud API)
- Sliding-window Multi-turn Chat Memory
- Guardrails & Strict Grounding in AnyFlip + Scraped Facts
"""
import json
import itertools
import math
import re
import time
import requests
from typing import List, Dict, Any, Tuple

from .config import models, paths
from .retriever import HybridRetriever
from .graph_engine import SongkhlaGraphEngine


SYSTEM_PROMPT = """
คุณคือ "น้องสิงขร" ผู้ช่วยแนะนำการท่องเที่ยวเมืองเก่าสงขลา

หน้าที่ของคุณคือตอบคำถามโดยใช้เฉพาะข้อมูลจาก Evidence/Context
ที่ระบบส่งมาให้เท่านั้น ห้ามใช้ความรู้ภายนอกหรือความรู้เดิมของโมเดล
เพื่อเติมข้อมูลที่ไม่มีในหลักฐาน

==================================================
1. กฎสูงสุด: Grounding
==================================================

ก่อนตอบทุกครั้ง ให้ตรวจสอบว่า Evidence รองรับสิ่งที่ผู้ใช้ถามจริงหรือไม่

Evidence ที่ Retriever ส่งมาอาจมีข้อมูลที่ไม่เกี่ยวข้องกับคำถาม
ดังนั้นห้ามถือว่าทุกข้อมูลใน Context สามารถนำมาตอบได้

ถ้า Evidence รองรับชัดเจน:
→ ตอบจาก Evidence

ถ้า Evidence รองรับเพียงบางส่วน:
→ ตอบเฉพาะส่วนที่รองรับ และแจ้งว่าส่วนที่เหลือไม่มีข้อมูลยืนยัน

ถ้า Evidence ไม่รองรับ:
→ ตอบอย่างกระชับว่าไม่มีข้อมูล

ถ้าไม่แน่ใจ:
→ ห้ามเดา

การตอบว่า "ไม่มีข้อมูล" ดีกว่าการสร้างคำตอบที่ไม่มีหลักฐานรองรับ

==================================================
2. Answer Scope — ตอบเฉพาะสิ่งที่ถาม
==================================================

ตอบเฉพาะข้อมูลที่จำเป็นต่อคำถามของผู้ใช้
ห้ามเพิ่มรายละเอียดอื่นเพียงเพราะรายละเอียดนั้นปรากฏใน Context

หลักการ:

ถาม 1 ข้อเท็จจริง
→ ตอบ 1 ข้อเท็จจริง

ถามหลายเงื่อนไข
→ ตอบเฉพาะเงื่อนไขที่ถาม

ถามให้เปรียบเทียบ
→ เปรียบเทียบเฉพาะสิ่งที่ถาม

ถามให้วางแผน
→ จึงสามารถตอบหลายขั้นตอนพร้อมเหตุผลได้

คำถามข้อเท็จจริงสั้น ๆ เช่น:
- เวลาเปิด-ปิด
- ราคา
- ค่าเข้า
- ที่ตั้ง
- ถนน
- เบอร์โทร
- ระยะทาง

ให้ตอบตรงคำถามภายใน 1 ประโยค ถ้าสามารถตอบได้ครบใน 1 ประโยค

ตัวอย่าง:

ผู้ใช้:
"ร้านไอติมโอ่งเปิดกี่โมง"

Evidence:
ร้านไอติมโอ่ง
เวลา 10:00–18:30 น.

ตอบ:
"ร้านไอติมโอ่งเปิด 10:00–18:30 น. ครับ"

ห้ามตอบ:
"ร้านไอติมโอ่งเปิด 10:00–18:30 น. ทุกวัน ตามข้อมูลที่ได้รับจากหลักฐานบริบทครับ"

เพราะผู้ใช้ไม่ได้ถามวันเปิดให้บริการ และ Evidence ไม่ได้ยืนยันคำว่า "ทุกวัน"

อีกตัวอย่าง:

ผู้ใช้:
"ร้านไอติมโอ่งราคาเท่าไหร่"

ตอบ:
"ราคา 20–30 บาทครับ"

ไม่ต้องบอกเวลาเปิด ถนน เมนู คะแนน หรือข้อมูลอื่นที่ไม่ได้ถาม

ห้ามใช้ข้อความเชิงระบบในคำตอบ เช่น:
- "ตามหลักฐานบริบท"
- "จาก Context"
- "จาก Retriever"
- "จากข้อมูลที่ระบบค้นคืนมา"
- "จาก Graph"
- "จากฐานข้อมูลที่ได้รับ"

ให้ตอบเหมือนผู้ช่วยท่องเที่ยวตามธรรมชาติ

==================================================
3. ห้าม Hallucination
==================================================

ห้ามสร้าง เดา หรือเติมข้อเท็จจริงที่ไม่มีอยู่ใน Evidence เช่น:

- สถานที่
- เส้นทาง
- ถนน
- ระยะทาง
- เวลาเดินทาง
- เวลาเปิด-ปิด
- วันเปิดให้บริการ
- ราคา
- ค่าเข้าชม
- เมนู
- หมายเลขโทรศัพท์
- พิกัด
- คะแนน
- สิ่งอำนวยความสะดวก
- ประวัติศาสตร์
- ความสัมพันธ์ระหว่างสถานที่
- ลำดับการเดินทาง

ห้ามใช้ความรู้ทั่วไปหรือความรู้เดิมของโมเดลเติมช่องว่าง
แม้ว่าคุณจะคิดว่ารู้คำตอบก็ตาม

สำคัญ:
ห้ามอนุมานข้อมูลใหม่จากข้อมูลอื่นโดยไม่มี Evidence รองรับโดยตรง

ตัวอย่าง:

Evidence:
"เปิด 10:00–18:30 น."

สามารถตอบ:
"เปิด 10:00–18:30 น."

ห้ามเพิ่ม:
"เปิดทุกวัน"
"เปิดวันจันทร์ถึงอาทิตย์"
"ไม่มีวันหยุด"

เว้นแต่ Evidence ระบุข้อมูลเหล่านั้นโดยตรง

==================================================
4. Exact Values
==================================================

เวลา ราคา ระยะทาง ค่าเข้า และตัวเลขทุกชนิด
ต้องใช้ค่าตาม Evidence ตรง ๆ

ถ้า Evidence ระบุ:
390 เมตร

ต้องตอบ:
390 เมตร

ห้ามเปลี่ยนเป็น:
ประมาณ 400 เมตร

ห้ามคำนวณตัวเลขใหม่เอง เว้นแต่ Context ส่งผลการคำนวณมาให้แล้ว

สำหรับระยะทางจาก latitude/longitude:
ถ้าระบบส่งค่าที่คำนวณด้วย Haversine มาแล้ว
ให้เรียกว่า:

"ระยะห่างโดยประมาณแบบเส้นตรง"

ห้ามเรียกว่า:
- ระยะเดิน
- ระยะขับรถ
- ระยะทางตามถนน
- เวลาเดินทาง

หาก Context ไม่ได้ให้ข้อมูลเหล่านั้น

==================================================
5. Entity Grounding
==================================================

ต้องตอบเกี่ยวกับ entity ที่ผู้ใช้ถามเท่านั้น

ถ้าผู้ใช้ถามสถานที่ A
แต่ Context มีเฉพาะ B และ C
ห้ามใช้ B หรือ C มาตอบแทน A

ห้ามถือว่าสถานที่สองแห่งเป็นสถานที่เดียวกันเพียงเพราะชื่อคล้ายกัน

ถ้าสถานที่ที่ถามไม่มี Evidence ให้ตอบ เช่น:

"ขออภัยครับ ตอนนี้ยังไม่มีข้อมูลเกี่ยวกับบ้านพรุ จึงยังไม่สามารถแนะนำเส้นทางได้ครับ"

ห้ามกล่าวถึงสถานที่อื่นที่ Retriever ส่งมาโดยไม่เกี่ยวข้อง

==================================================
6. Known Entity แต่ไม่มี Attribute
==================================================

ต้องแยกให้ออกระหว่าง:

A. ไม่มีข้อมูลเกี่ยวกับสถานที่นั้นเลย

กับ

B. มีข้อมูลสถานที่ แต่ไม่มีข้อมูล attribute ที่ผู้ใช้ถาม

ตัวอย่าง:

มีข้อมูลร้านไอติมโอ่ง
แต่ไม่มีข้อมูลที่จอดรถ

ผู้ใช้:
"ร้านไอติมโอ่งมีที่จอดรถไหม"

ตอบ:
"ตอนนี้ยังไม่มีข้อมูลยืนยันเรื่องที่จอดรถของร้านไอติมโอ่งครับ"

ห้ามตอบว่า:
"ไม่มีข้อมูลเกี่ยวกับร้านไอติมโอ่ง"

==================================================
7. Multi-Entity
==================================================

สำหรับคำถามหลายสถานที่
ให้ใช้เฉพาะสถานที่ที่ผู้ใช้ระบุ
หรือสถานที่ที่ระบบ resolve/กำหนดเป็น allowed_places

ห้ามเพิ่มสถานที่อื่นจาก Retrieval เข้ามาเอง

ถ้าผู้ใช้ถาม:
"ร้านไอติมโอ่งกับบ้านจีน 300 ปีไกลกันไหม"

ต้องพิจารณาเฉพาะ:
- ร้านไอติมโอ่ง
- บ้านจีน 300 ปี

ห้ามเพิ่มโรงสีแดงหรือสถานที่อื่น

หากสถานที่ใดสถานที่หนึ่งไม่มีข้อมูล
ห้ามนำสถานที่อื่นมาแทน

==================================================
8. Recommendation
==================================================

คำแนะนำต้องผ่านทุกเงื่อนไขที่ผู้ใช้กำหนด

ตัวอย่าง:

"แนะนำของหวานงบไม่เกิน 30 บาท"

สถานที่ที่จะนำมาตอบต้องมี Evidence รองรับว่า:
1. เป็นของหวาน/ของกินเล่นที่ตรงคำถาม
2. ราคาอยู่ภายในงบที่กำหนด

ถ้าพิสูจน์เงื่อนไขไม่ได้ครบ:
→ ห้ามแนะนำ

ห้ามนำร้านอาหารคาวมาตอบคำถามร้านของหวาน
เพียงเพราะ Retrieval จัดอันดับร้านนั้นสูง

ถ้ามีหลายตัวเลือกที่ผ่านเงื่อนไข:
→ สามารถแนะนำหลายตัวเลือกได้

ถ้าไม่มีตัวเลือกที่พิสูจน์ได้:
→ บอกว่าไม่มีข้อมูลเพียงพอ

==================================================
9. Route / Itinerary
==================================================

ถ้าระบบส่ง route/lำดับการเดินทางที่คำนวณไว้แล้ว:
→ ใช้ลำดับนั้น
→ ห้ามคิดลำดับใหม่เอง

ถ้าระบบส่งระยะทางมา:
→ ใช้ค่าที่ส่งมา
→ ห้ามคำนวณหรือเปลี่ยนค่าเอง

สามารถอธิบายเหตุผลของลำดับได้เฉพาะเมื่อ Evidence รองรับ เช่น:

- เวลาเปิด-ปิด
- ประเภทสถานที่
- ระยะทาง
- ความสัมพันธ์ที่ระบุใน Context
- เงื่อนไขของผู้ใช้

ห้ามสร้างเหตุผลขึ้นมาเพื่อให้คำตอบดูสมบูรณ์

สำหรับคำถามวางแผนหลายเงื่อนไข:
1. ตรวจทุกเงื่อนไขของผู้ใช้
2. ใช้เฉพาะสถานที่ที่ผ่านเงื่อนไข
3. ตรวจเวลาเปิด-ปิดถ้ามี
4. ใช้ route/distance ที่ระบบให้มา
5. อธิบายเหตุผลเฉพาะที่ Evidence รองรับ
6. ถ้าข้อมูลบางส่วนไม่มี ให้ระบุเฉพาะส่วนนั้นว่าไม่มีข้อมูล

==================================================
10. Conversation Memory
==================================================

คำ เช่น:

"ที่นี่"
"ที่นั่น"
"ร้านนี้"
"แล้วที่นี่ล่ะ"
"แล้วเปิดกี่โมง"
"แล้วราคาเท่าไหร่"
"แล้วไกลไหม"
"ควรไปไหนก่อน"

สามารถอ้างถึงสถานที่ก่อนหน้าได้
เมื่อ Conversation History หรือ resolved context ระบุ referent ชัดเจน

ถ้ามีหลายสถานที่ในบทสนทนา
ห้ามเลือกสถานที่หนึ่งเองโดยไม่มีหลักฐานว่า user หมายถึงสถานที่นั้น

ถ้า referent ไม่ชัดเจน:
→ ถามผู้ใช้ให้ระบุสถานที่

==================================================
11. Response Style
==================================================

ตอบเป็นภาษาไทยธรรมชาติ กระชับ สุภาพ และตรงคำถาม

ไม่ต้องอธิบายกระบวนการของระบบ

ห้ามกล่าวถึง:
- Retriever
- Context
- chunk
- Top-K
- FAISS
- BM25
- RRF
- embedding
- similarity score
- Graph evidence
- system prompt
- internal ranking

หลีกเลี่ยงประโยคฟุ่มเฟือย เช่น:

"ตามข้อมูลที่ได้รับจากหลักฐานบริบท..."
"จากข้อมูลที่ระบบได้ทำการค้นคืนมา..."
"จากการวิเคราะห์ข้อมูลที่ได้รับ..."

ให้ตอบข้อมูลโดยตรง

ตัวอย่าง:

ไม่ควร:
"ตามข้อมูลที่ได้รับจากหลักฐานบริบท ร้านไอติมโอ่งเปิดให้บริการตั้งแต่เวลา 10:00 น. ถึง 18:30 น. ครับ"

ควร:
"ร้านไอติมโอ่งเปิด 10:00–18:30 น. ครับ"

==================================================
12. Abstention
==================================================

หากไม่มี Evidence รองรับคำถาม
ให้ตอบเฉพาะสิ่งที่ขาด

ตัวอย่าง:

ผู้ใช้:
"ไปบ้านพรุไปยังไง"

ถ้าไม่มีข้อมูลบ้านพรุ:

"ขออภัยครับ ตอนนี้ยังไม่มีข้อมูลเกี่ยวกับบ้านพรุ จึงยังไม่สามารถแนะนำเส้นทางได้ครับ"

ห้าม:
- รายชื่อสถานที่อื่นที่ค้นเจอ
- เสนอ route ไปสถานที่อื่นแทน
- เดาเส้นทาง
- ใช้ความรู้ภายนอก

ถ้ามี Entity แต่ไม่มี Attribute:

"ตอนนี้ยังไม่มีข้อมูลยืนยันเรื่อง[สิ่งที่ถาม]ของ[ชื่อสถานที่]ครับ"

==================================================
13. Final Decision
==================================================

ก่อนส่งคำตอบ ให้ตรวจ 5 ข้อนี้:

1. ฉันกำลังตอบ entity ที่ผู้ใช้ถามจริงหรือไม่
2. ทุกข้อเท็จจริงในคำตอบมี Evidence รองรับหรือไม่
3. ตัวเลขทั้งหมดตรงกับ Evidence หรือไม่
4. ฉันเพิ่มข้อมูลที่ผู้ใช้ไม่ได้ถามโดยไม่จำเป็นหรือไม่
5. ฉันกล่าวถึงสถานที่ที่ไม่เกี่ยวข้องหรือไม่

ถ้าข้อ 1-3 ไม่ผ่าน:
→ ห้ามตอบข้อมูลนั้น

ถ้าข้อ 4-5 เป็น "ใช่":
→ ตัดข้อมูลส่วนนั้นออกก่อนตอบ

เป้าหมายคือ:
"ตอบให้น้อยที่สุด แต่ครบสิ่งที่ผู้ใช้ถาม และทุกคำตอบต้องมีหลักฐานรองรับ"
"""

class SongkhlaRAGEngine:
    def __init__(self):
        print("=" * 60)
        print("🚀 Initializing Songkhla Old Town Hybrid GraphRAG Engine...")
        print("=" * 60)
        self.graph_engine = SongkhlaGraphEngine()
        self.retriever = HybridRetriever(graph_engine=self.graph_engine)
        self.groq_api_key = models.groq_api_key
        self.groq_model = models.groq_model
        self.ollama_base_url = models.ollama_base_url
        self.local_model = models.primary_local_llm
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    def _get_known_place_names(self) -> List[str]:
        """Load canonical place names for resolving conversational references."""
        cached = getattr(self, "_known_place_names", None)
        if cached is not None:
            return cached

        names = set()
        try:
            with open(paths.facts_path, "r", encoding="utf-8") as f:
                for place in json.load(f):
                    name = str(place.get("name", "")).strip()
                    if name:
                        names.add(name)
                        names.add(name.split("(", 1)[0].strip())
        except (OSError, ValueError, TypeError):
            pass

        self._known_place_names = sorted(
            (name for name in names if name), key=len, reverse=True
        )
        return self._known_place_names

    def _get_place_aliases(self) -> List[Tuple[str, str]]:
        """Return longest-first (alias, canonical name) pairs from place facts."""
        cached = getattr(self, "_place_aliases", None)
        if cached is not None:
            return cached

        alias_to_canonical = {}
        generic_prefixes = {"สงขลา", "ร้าน", "บ้าน", "โรงแรม"}
        try:
            with open(paths.facts_path, "r", encoding="utf-8") as f:
                places = json.load(f)
            for place in places:
                full_name = re.sub(r"\s+", " ", str(place.get("name", "")).strip())
                canonical = full_name.split("(", 1)[0].strip()
                if not canonical:
                    continue
                aliases = {full_name, canonical}
                first_phrase = canonical.split(" ", 1)[0]
                if len(first_phrase) >= 5 and first_phrase not in generic_prefixes:
                    aliases.add(first_phrase)
                for alias in aliases:
                    if alias:
                        alias_to_canonical.setdefault(alias, canonical)
        except (OSError, ValueError, TypeError):
            for name in self._get_known_place_names():
                alias_to_canonical.setdefault(name, name)

        self._place_aliases = sorted(
            alias_to_canonical.items(), key=lambda item: len(item[0]), reverse=True
        )
        return self._place_aliases

    def _get_place_records(self) -> Dict[str, Dict[str, Any]]:
        """Load place facts keyed by canonical name without mutating the source data."""
        cached = getattr(self, "_place_records", None)
        if cached is not None:
            return cached
        records = {}
        try:
            with open(paths.facts_path, "r", encoding="utf-8") as f:
                for place in json.load(f):
                    canonical = str(place.get("name", "")).split("(", 1)[0].strip()
                    if canonical:
                        records[canonical] = place
        except (OSError, ValueError, TypeError):
            pass
        self._place_records = records
        return records

    def _extract_place_entities(self, text: str) -> List[str]:
        """Extract unique canonical places in their mention order."""
        matches = []
        for alias, canonical in self._get_place_aliases():
            start = text.find(alias)
            if start >= 0:
                matches.append((start, -len(alias), canonical))

        entities = []
        seen = set()
        for _, _, canonical in sorted(matches):
            if canonical not in seen:
                seen.add(canonical)
                entities.append(canonical)
        return entities

    @staticmethod
    def _is_discovery_query(query: str) -> bool:
        discovery_markers = (
            "มีที่เที่ยวอะไร", "แนะนำร้านอาหาร", "แนะนำของหวาน", "เที่ยวไหนดี",
            "ช่วยวางแผนเที่ยว", "มีอะไรแนะนำ", "ร้านอาหารแนะนำ", "ที่เที่ยวแนะนำ",
        )
        return any(marker in query for marker in discovery_markers)

    def _canonical_subject(self, text: str):
        entities = self._extract_place_entities(text)
        if entities:
            return entities[0]
        subject = re.sub(r"^[\s\"'“”]+|[\s\"'“”?!]+$", "", text).strip()
        subject = re.sub(r"^(?:อยากรู้(?:ว่า)?|ขอข้อมูล|เกี่ยวกับ)\s*", "", subject).strip()
        generic_subjects = {
            "ที่เที่ยว", "สถานที่ท่องเที่ยว", "ร้านอาหาร", "ของหวาน", "ร้านของหวาน",
            "คาเฟ่", "โรงแรม", "ที่พัก", "ร้าน", "สถานที่", "แถวนี้", "เมืองเก่าสงขลา",
        }
        return None if not subject or subject in generic_subjects else subject

    def _requested_subjects(self, query: str) -> Dict[str, List[str]]:
        """Extract only clear named subjects; generic discovery remains unblocked."""
        if self._is_discovery_query(query):
            return {"known": [], "unknown": []}

        subjects = []
        pair = re.search(
            r"(.+?)กับ(.+?)(?:ไกลกันไหม|ห่างกัน(?:ไหม|เท่าไหร่)?|อยู่ห่าง.*?กี่เมตร|ที่ไหนใกล้กว่ากัน)",
            query,
        )
        if pair:
            subjects.extend((pair.group(1), pair.group(2)))
        else:
            patterns = (
                r"^(?:อยาก)?ไป\s*(.+?)\s*(?:ต้อง)?\s*(?:ไปยังไง|เดินทางยังไง|ยังไง|อย่างไร|ทางไหน|อยู่ไหน)$",
                r"^(.+?)\s*(?:เปิดกี่โมง|ปิดกี่โมง|ราคาเท่าไหร่|ราคาเท่าไร|มีที่จอดรถไหม|อยู่ที่ไหน|อยู่ไหน)$",
            )
            for pattern in patterns:
                match = re.search(pattern, query.strip(), flags=re.IGNORECASE)
                if match:
                    subjects.append(match.group(1))
                    break
        if not subjects:
            subjects.extend(self._extract_place_entities(query))

        known, unknown = [], []
        records = self._get_place_records()
        for raw_subject in subjects:
            subject = self._canonical_subject(raw_subject)
            if not subject:
                continue
            if subject in records:
                if subject not in known:
                    known.append(subject)
            elif subject not in unknown:
                unknown.append(subject)
        return {"known": known, "unknown": unknown}

    @staticmethod
    def _requested_attribute(query: str):
        attributes = (
            ("parking", ("ที่จอดรถ", "จอดรถ")),
            ("opening_hours", ("เปิดกี่โมง", "ปิดกี่โมง", "เวลาเปิด", "เวลาปิด")),
            ("price", ("ราคาเท่าไหร่", "ราคาเท่าไร", "ค่าเข้า", "ค่าลิฟต์", "กี่บาท", "ราคา")),
            ("route", ("ไปยังไง", "เดินทาง", "ทางไป")),
        )
        for key, markers in attributes:
            if any(marker in query for marker in markers):
                return key
        return None

    @staticmethod
    def _attribute_label(attribute: str) -> str:
        return {
            "parking": "ที่จอดรถ",
            "opening_hours": "เวลาเปิด-ปิด",
            "price": "ราคา",
            "route": "เส้นทาง",
        }.get(attribute, "ข้อมูลที่ถาม")

    def _evidence_has_attribute(
        self, entity: str, attribute: str, chunks: List[Dict[str, Any]]
    ) -> bool:
        markers = {
            "parking": ("ที่จอดรถ", "จอดรถ", "parking"),
            "opening_hours": ("เวลาเปิด", "เปิดเวลา", "open_hours"),
            "price": ("ราคา", "บาท", "price_range"),
            "route": ("ที่ตั้ง", "ถนน", "พิกัด", "lat", "lon", "maps"),
        }.get(attribute, ())
        evidence = json.dumps(self._get_place_records().get(entity, {}), ensure_ascii=False)
        for chunk in chunks:
            chunk_text = json.dumps(chunk, ensure_ascii=False)
            if entity in chunk_text:
                evidence += " " + chunk_text
        return any(marker.lower() in evidence.lower() for marker in markers)

    def _fact_attribute_answer(self, entity: str, attribute: str):
        """Recover a known factual attribute when generation abstains incorrectly."""
        record = self._get_place_records().get(entity, {})
        if attribute == "price" and record.get("price_range"):
            return f"{entity}มีข้อมูลราคา: {record['price_range']}ครับ"
        if attribute == "opening_hours" and record.get("open_hours"):
            return f"{entity}เปิดให้บริการเวลา {record['open_hours']}ครับ"
        return None

    def _guardrail_result(
        self, query, retrieval_query, answer, chunks, start_time,
        user_id, status, reason, entities, mode,
    ) -> Dict[str, Any]:
        if user_id:
            self.sessions.setdefault(user_id, []).extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ])
            self.sessions[user_id] = self.sessions[user_id][-6:]
        return {
            "query": query,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "provider": "deterministic_guardrail",
            "model": None,
            "latency_seconds": round(time.time() - start_time, 2),
            "chunks_count": len(chunks),
            "sources": [],
            "mode": mode,
            "allowed_places": [],
            "route_plan": None,
            "distance_plan": None,
            "guardrail_status": status,
            "guardrail_reason": reason,
            "resolved_entities": entities,
        }

    @staticmethod
    def _is_route_question(query: str) -> bool:
        route_markers = (
            "ควรไปไหนก่อน", "ควรไปที่ไหนก่อน", "ก่อน-หลัง", "ก่อนหลัง",
            "ไปที่ไหนก่อน", "เรียงให้หน่อย", "ไปต่อไหนดี", "แล้วไปไหนต่อ",
            "ไปไหนต่อ", "ลำดับ", "เส้นทาง"
        )
        return any(marker in query for marker in route_markers)

    @staticmethod
    def _is_distance_question(query: str) -> bool:
        distance_markers = (
            "ไกลกันไหม", "ห่างกัน", "อยู่ห่าง", "กี่เมตร",
            "ระยะห่าง", "ระยะทาง", "ที่ไหนใกล้กว่ากัน"
        )
        return any(marker in query for marker in distance_markers)

    @staticmethod
    def _haversine_meters(place_a: Dict[str, Any], place_b: Dict[str, Any]):
        """Calculate straight-line metres from existing lat/lon fact fields."""
        try:
            lat1, lon1 = float(place_a["lat"]), float(place_a["lon"])
            lat2, lon2 = float(place_b["lat"]), float(place_b["lon"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
            return None
        if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return None
        lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        value = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return round(6371000 * 2 * math.asin(math.sqrt(value)))

    def resolve_pairwise_distance(
        self,
        place_a: str,
        place_b: str,
        chunks: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Resolve explicit evidence first, then straight-line Haversine distance."""
        graph = getattr(getattr(self, "graph_engine", None), "G", None)
        if graph is not None:
            for source, target in ((place_a, place_b), (place_b, place_a)):
                if graph.has_edge(source, target):
                    edge = graph.get_edge_data(source, target) or {}
                    distance = edge.get("distance_m")
                    if isinstance(distance, (int, float)) and distance >= 0:
                        meters = round(distance)
                        return {
                            "meters": meters,
                            "source": "explicit",
                            "description": f"ระยะทางตามหลักฐาน {meters} เมตร",
                        }

        for chunk in chunks or []:
            evidence_items = [chunk.get("content", "")]
            evidence_items.extend(
                item.get("content", "") if isinstance(item, dict) else str(item)
                for item in chunk.get("graph_evidence", [])
            )
            for content in evidence_items:
                for segment in re.split(r"[\n;]", str(content)):
                    if place_a not in segment or place_b not in segment:
                        continue
                    match = re.search(r"(\d+(?:\.\d+)?)\s*(กิโลเมตร|เมตร)", segment)
                    if match:
                        value = float(match.group(1))
                        meters = round(value * 1000 if match.group(2) == "กิโลเมตร" else value)
                        return {
                            "meters": meters,
                            "source": "explicit",
                            "description": f"ระยะทางตามหลักฐาน {meters} เมตร",
                        }

        records = self._get_place_records()
        meters = self._haversine_meters(records.get(place_a, {}), records.get(place_b, {}))
        if meters is not None:
            return {
                "meters": meters,
                "source": "haversine",
                "description": f"ระยะห่างโดยประมาณแบบเส้นตรง {meters} เมตร",
            }
        return {"meters": None, "source": "unknown", "description": "ไม่ทราบระยะห่าง"}

    def _build_route_plan(
        self,
        places: List[str],
        chunks: List[Dict[str, Any]],
        query: str = "",
    ) -> Dict[str, Any]:
        """Order only requested places using constraints, roles, hours, and distance."""
        allowed_places = list(dict.fromkeys(places))
        records = self._get_place_records()

        def role_rank(place_name):
            category = str(records.get(place_name, {}).get("category", ""))
            if any(label in category for label in (
                "ประวัติศาสตร์", "วัฒนธรรม", "พิพิธภัณฑ์", "ศิลปะ",
                "จุดชมวิว", "สถานที่ท่องเที่ยว", "กิจกรรม",
            )):
                return 0
            if "อาหารคาว" in category:
                return 1
            if any(label in category for label in ("ของหวาน", "คาเฟ่", "เครื่องดื่ม")):
                return 2
            return 1

        def opening_minutes(place_name):
            hours = str(records.get(place_name, {}).get("open_hours", ""))
            match = re.search(r"(\d{1,2}):([0-5]\d)", hours)
            return int(match.group(1)) * 60 + int(match.group(2)) if match else None

        def constrained_place(pattern):
            for alias, canonical in self._get_place_aliases():
                if canonical not in allowed_places:
                    continue
                if re.search(pattern.format(alias=re.escape(alias)), query):
                    return canonical
            return None

        explicit_start = constrained_place(
            r"(?:เริ่ม(?:ต้น)?จาก|ไป)\s*(?:ที่)?\s*{alias}\s*(?:ก่อน)?"
        )
        explicit_end = constrained_place(r"(?:ปิดท้ายที่|จบที่)\s*{alias}")
        if explicit_start == explicit_end:
            explicit_end = None

        distance_cache = {}
        for first, second in itertools.combinations(allowed_places, 2):
            distance_cache[frozenset((first, second))] = self.resolve_pairwise_distance(
                first, second, chunks
            )

        def route_score(route):
            ranks = [role_rank(place) for place in route]
            role_inversions = sum(
                ranks[index] > ranks[later]
                for index in range(len(ranks))
                for later in range(index + 1, len(ranks))
            )
            opening_inversions = 0
            for index in range(len(route)):
                for later in range(index + 1, len(route)):
                    first_open = opening_minutes(route[index])
                    second_open = opening_minutes(route[later])
                    if (
                        ranks[index] == ranks[later]
                        and first_open is not None and second_open is not None
                        and first_open > second_open
                    ):
                        opening_inversions += 1
            leg_distances = [
                distance_cache[frozenset((first, second))]["meters"]
                for first, second in zip(route, route[1:])
            ]
            unknown_legs = sum(distance is None for distance in leg_distances)
            total_distance = sum(distance or 0 for distance in leg_distances)
            return (
                role_inversions,
                opening_inversions,
                unknown_legs,
                total_distance,
                tuple(route),
            )

        candidates = [
            route for route in itertools.permutations(allowed_places)
            if (not explicit_start or route[0] == explicit_start)
            and (not explicit_end or route[-1] == explicit_end)
        ]
        ordered = list(min(candidates, key=route_score))
        legs = []
        for current, next_place in zip(ordered, ordered[1:]):
            legs.append({
                "from": current,
                "to": next_place,
                "distance": distance_cache[frozenset((current, next_place))],
            })

        decisions = []
        if explicit_start:
            decisions.append(f"ผู้ใช้ระบุให้เริ่มจาก{explicit_start}")
        if explicit_end:
            decisions.append(f"ผู้ใช้ระบุให้ปิดท้ายที่{explicit_end}")
        attraction_places = [place for place in ordered if role_rank(place) == 0]
        if len(attraction_places) >= 2:
            categories = [records[place].get("category", "") for place in attraction_places]
            unique_categories = list(dict.fromkeys(categories))
            if len(unique_categories) == 1:
                decisions.append(
                    f"ทั้ง{' และ '.join(attraction_places)}มีหมวดข้อมูลเป็น"
                    f"{unique_categories[0]} จึงจัดไว้ต่อเนื่องกัน"
                )
            else:
                decisions.append(
                    f"จัด{' และ '.join(attraction_places)}ไว้ต่อเนื่องกันตามหมวดข้อมูล "
                    + " / ".join(unique_categories)
                )
        if not explicit_start and len(attraction_places) >= 2:
            first, second = attraction_places[:2]
            first_open = opening_minutes(first)
            second_open = opening_minutes(second)
            if first_open is not None and second_open is not None and first_open < second_open:
                decisions.append(
                    f"{first}เปิดก่อน ({records[first].get('open_hours')}) เมื่อเทียบกับ"
                    f"{second} ({records[second].get('open_hours')})"
                )
        final_category = str(records.get(ordered[-1], {}).get("category", ""))
        if role_rank(ordered[-1]) == 2 and not explicit_end:
            decisions.append(
                f"ปิดท้ายที่{ordered[-1]} เพราะข้อมูลจัดเป็นหมวด{final_category}"
            )

        if len(ordered) == 2:
            answer = f"แนะนำให้ไป{ordered[0]}ก่อน แล้วค่อยไป{ordered[1]}ครับ\n"
            if decisions:
                answer += f"เหตุผล: {' '.join(decisions)}\n"
            if legs[0]["distance"]["meters"] is None:
                answer += "ยังไม่มีหลักฐานหรือพิกัดเพียงพอสำหรับยืนยันระยะห่างระหว่างสองแห่ง"
            else:
                answer += f"ทั้งสองแห่งมี{legs[0]['distance']['description']}"
        else:
            lines = ["แนะนำลำดับดังนี้ครับ"]
            lines.extend(f"{index}. {place}" for index, place in enumerate(ordered, 1))
            known_descriptions = [
                f"{leg['from']} → {leg['to']}: {leg['distance']['description']}"
                for leg in legs if leg["distance"]["meters"] is not None
            ]
            reason = " ".join(decisions) if decisions else "จัดลำดับจากข้อมูลประเภทสถานที่ เวลาเปิด และระยะห่างที่มี"
            if known_descriptions:
                reason += "\n" + "\n".join(known_descriptions)
            lines.extend(["", f"เหตุผล: {reason}"])
            answer = "\n".join(lines)
        return {
            "allowed_places": allowed_places,
            "ordered_places": ordered,
            "legs": legs,
            "decisions": decisions,
            "answer": answer,
        }

    @classmethod
    def _is_follow_up_query(cls, query: str) -> bool:
        if cls._is_route_question(query) or cls._is_distance_question(query):
            return True
        return bool(re.match(
            r"^(?:แล้ว(?:ล่ะ|ละ)?|ส่วน|ที่นั่น|ที่นี่|ที่นี้|ร้านนี้)(?:\s*|$)",
            query
        ))

    def _has_self_relation_claim(self, answer: str) -> bool:
        """Detect a route answer claiming that a canonical place is near itself."""
        for alias, _ in self._get_place_aliases():
            escaped = re.escape(alias)
            if re.search(
                rf"{escaped}.{{0,50}}(?:ใกล้|NEAR(?:BY)?).{{0,50}}{escaped}",
                answer,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    @staticmethod
    def _collapse_unsupported_answer(answer: str) -> str:
        """Prevent an abstention from continuing with unrelated recommendations."""
        abstention_markers = (
            "ยังไม่พบข้อมูล", "ไม่มีข้อมูล", "ข้อมูลไม่เพียงพอ",
            "ไม่สามารถยืนยัน", "หลักฐานไม่เพียงพอ",
        )
        if any(marker in answer for marker in abstention_markers):
            return "ขออภัยครับ จากข้อมูลที่มีตอนนี้ยังไม่พบข้อมูลที่เกี่ยวข้องครับ"
        return answer

    def resolve_retrieval_query(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """Resolve referential follow-ups with canonical entities from the last turn."""
        clean_query = query.strip()
        if not history or not self._get_place_aliases():
            return clean_query
        if self._extract_place_entities(clean_query):
            return clean_query
        if not self._is_follow_up_query(clean_query):
            return clean_query

        previous_user = next(
            (item.get("content", "") for item in reversed(history)
             if item.get("role") == "user"),
            ""
        )
        previous_assistant = next(
            (item.get("content", "") for item in reversed(history)
             if item.get("role") == "assistant"),
            ""
        )
        user_entities = self._extract_place_entities(previous_user)
        assistant_entities = self._extract_place_entities(previous_assistant)
        # Prefer the user's explicit topic. Use the assistant when it introduced
        # the place (for example after a broad recommendation request).
        entities = user_entities or assistant_entities
        if not entities:
            return clean_query

        follow_up = re.sub(
            r"^(?:แล้ว(?:ล่ะ|ละ)?|ส่วน)\s*", "", clean_query
        ).strip()
        follow_up = re.sub(
            r"^(?:ร้านนี้|ที่นั่น|ที่นี่|ที่นี้)\s*", "", follow_up
        ).strip()
        return " ".join([*entities, follow_up] if follow_up else entities)

    def route_model(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Adaptive Model Router:
        - Default to Local LLM (Ollama) for fast, concise, offline-first execution of factual queries.
        - Route to Cloud API (Groq) for multi-day itinerary synthesis, complex planning, or multi-constraint reasoning.
        """
        complex_keywords = [
            "จัดทริป", "วางแผนเที่ยว", "จัดตาราง", "2 วัน", "3 วัน", "ทั้งวัน",
            "เปรียบเทียบ", "ข้อดีข้อเสีย", "วิเคราะห์", "คำนวณงบ", "งบประมาณ"
        ]
        
        is_complex = any(kw in query for kw in complex_keywords) or (len(query.strip()) > 50 and "และ" in query)
        
        if is_complex and self.groq_api_key:
            self._last_route_reason = "complex planning/synthesis query"
            return "groq", self.groq_model
        if is_complex:
            self._last_route_reason = "complex query; Groq is not configured, using local"
        else:
            self._last_route_reason = "factual/simple query"
        return "ollama", self.local_model

    @staticmethod
    def _log_route(query: str, provider: str, model_name: str, reason: str) -> None:
        """Print routing observability to the server terminal only."""
        is_cloud = provider == "groq"
        print("-" * 50, flush=True)
        print("[LLM ROUTER]", flush=True)
        print(f"Query: {query}", flush=True)
        print(f"Provider: {'CLOUD' if is_cloud else 'LOCAL'}", flush=True)
        print(f"Backend: {'Groq' if is_cloud else 'Ollama'}", flush=True)
        print(f"Model: {model_name}", flush=True)
        print(f"Reason: {reason}", flush=True)
        print("Fallback: NO", flush=True)
        print("-" * 50, flush=True)

    @staticmethod
    def _log_fallback(
        original_provider: str,
        fallback_provider: str,
        reason: str,
        succeeded: bool
    ) -> None:
        """Log a safe fallback category without response bodies or credentials."""
        labels = {"groq": "CLOUD/Groq", "ollama": "LOCAL/Ollama"}
        print("[LLM FALLBACK]", flush=True)
        print(f"Original provider: {labels[original_provider]}", flush=True)
        print(f"Fallback provider: {labels[fallback_provider]}", flush=True)
        print(f"Reason: {reason}", flush=True)
        print(f"Result: {'SUCCEEDED' if succeeded else 'FAILED'}", flush=True)

    def call_ollama(self, messages: List[Dict[str, str]], model_name: str) -> str:
        """Invokes Local LLM via Ollama API."""
        try:
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2, "top_p": 0.9}
            }
            res = requests.post(f"{self.ollama_base_url}/api/chat", json=payload, timeout=45)
            if res.status_code == 200:
                return res.json().get("message", {}).get("content", "").strip()
            return f"Ollama Error (Status {res.status_code}): {res.text}"
        except Exception as e:
            return f"Local LLM Error: {e}"

    def call_groq(self, messages: List[Dict[str, str]], model_name: str) -> str:
        """Invokes Cloud API LLM via Groq API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 1000
            }
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"].strip()
            return f"Groq Error (Status {res.status_code}): {res.text}"
        except Exception as e:
            return f"Cloud API Error: {e}"

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]]) -> str:
        """Render retrieved text and graph evidence into the LLM context."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title") or f"เอกสารที่ {i}"
            chunk_id = chunk.get("chunk_id")
            source_page = chunk.get("source_page")
            raw_content = chunk.get("content")
            content = str(raw_content).strip() if raw_content is not None else ""
            is_graph_only = chunk.get("category") == "Knowledge Graph"
            evidence_blocks = []
            seen_contents = set()

            if content and not is_graph_only:
                text_meta = [f"แหล่งข้อมูล: {title}"]
                if chunk_id:
                    text_meta.append(f"Chunk ID: {chunk_id}")
                if source_page is not None:
                    text_meta.append(f"หน้า: {source_page}")
                evidence_blocks.append(
                    "[หลักฐานข้อความ]\n"
                    + "\n".join(text_meta)
                    + f"\nเนื้อหา:\n{content}"
                )
                seen_contents.add(content)

            graph_evidence = chunk.get("graph_evidence", [])
            if not isinstance(graph_evidence, list):
                graph_evidence = []

            if is_graph_only and content:
                graph_evidence = [{
                    "chunk_id": chunk_id,
                    "title": title,
                    "content": content
                }, *graph_evidence]

            # Support hybrid results created before graph_evidence was added.
            raw_legacy_graph = chunk.get("graph_subgraph")
            legacy_graph_content = (
                str(raw_legacy_graph).strip() if raw_legacy_graph is not None else ""
            )
            if legacy_graph_content:
                graph_evidence = [*graph_evidence, {
                    "title": title,
                    "content": legacy_graph_content
                }]

            for graph_item in graph_evidence:
                if isinstance(graph_item, str):
                    graph_title = "Knowledge Graph"
                    graph_chunk_id = None
                    graph_content = graph_item.strip()
                elif isinstance(graph_item, dict):
                    graph_title = graph_item.get("title") or "Knowledge Graph"
                    graph_chunk_id = graph_item.get("chunk_id")
                    raw_graph_content = graph_item.get("content")
                    graph_content = (
                        str(raw_graph_content).strip() if raw_graph_content is not None else ""
                    )
                else:
                    continue

                if not graph_content or graph_content in seen_contents:
                    continue

                graph_meta = [f"แหล่งข้อมูลกราฟ: {graph_title}"]
                if graph_chunk_id:
                    graph_meta.append(f"Graph Chunk ID: {graph_chunk_id}")
                evidence_blocks.append(
                    "[หลักฐานกราฟ]\n"
                    + "\n".join(graph_meta)
                    + f"\nความสัมพันธ์ที่ค้นคืน:\n{graph_content}"
                )
                seen_contents.add(graph_content)

            if evidence_blocks:
                context_parts.append(
                    f"--- [หลักฐานลำดับที่ {i}] ---\n" + "\n\n".join(evidence_blocks)
                )

        return "\n\n".join(context_parts)

    def generate(
        self,
        query: str,
        mode: str = "hybrid",
        target_llm: str = None,
        user_id: str = "default_user",
        top_k: int = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG generation.
        """
        start_time = time.time()

        history = self.sessions.get(user_id, [])[-4:]  # Last 2 turns
        retrieval_query = self.resolve_retrieval_query(query, history)

        # 1. Retrieve Context
        chunks = self.retriever.retrieve(retrieval_query, top_k=top_k, mode=mode)

        subjects = self._requested_subjects(retrieval_query)
        requested_attribute = self._requested_attribute(query)
        if subjects["unknown"]:
            unknown_text = " และ ".join(subjects["unknown"])
            if subjects["known"] and self._is_distance_question(query):
                known_text = " และ ".join(subjects["known"])
                answer = (
                    f"ตอนนี้ระบบมีข้อมูล{known_text} แต่ยังไม่มีข้อมูลตำแหน่งของ{unknown_text} "
                    "จึงยังไม่สามารถคำนวณระยะห่างระหว่างสองสถานที่ได้ครับ"
                )
            elif requested_attribute == "route":
                answer = (
                    f"ขออภัยครับ ตอนนี้ระบบยังไม่มีข้อมูลเกี่ยวกับ{unknown_text} "
                    f"จึงยังไม่สามารถแนะนำเส้นทางไป{unknown_text}ได้ครับ"
                )
            elif requested_attribute == "opening_hours":
                answer = (
                    f"ขออภัยครับ ตอนนี้ระบบยังไม่มีข้อมูลเกี่ยวกับ{unknown_text} "
                    "จึงยังไม่สามารถยืนยันเวลาเปิด-ปิดได้ครับ"
                )
            elif requested_attribute == "price":
                answer = (
                    f"ขออภัยครับ ตอนนี้ระบบยังไม่มีข้อมูลเกี่ยวกับ{unknown_text} "
                    "จึงยังไม่สามารถยืนยันราคาได้ครับ"
                )
            else:
                answer = f"ขออภัยครับ ตอนนี้ระบบยังไม่มีข้อมูลเกี่ยวกับ{unknown_text}ครับ"
            print(
                f"[Guardrail] status=ABSTAIN unknown_entity={unknown_text!r}", flush=True
            )
            return self._guardrail_result(
                query, retrieval_query, answer, chunks, start_time, user_id,
                "ABSTAIN", "unknown_entity", subjects["known"], mode,
            )

        if (
            len(subjects["known"]) == 1
            and requested_attribute
            and not self._evidence_has_attribute(
                subjects["known"][0], requested_attribute, chunks
            )
        ):
            entity = subjects["known"][0]
            label = self._attribute_label(requested_attribute)
            answer = f"จากข้อมูลที่มี ยังไม่มีข้อมูลยืนยันเรื่อง{label}ของ{entity}ครับ"
            print(
                f"[Guardrail] status=PARTIAL missing_attribute={requested_attribute!r}",
                flush=True,
            )
            return self._guardrail_result(
                query, retrieval_query, answer, chunks, start_time, user_id,
                "PARTIAL", f"missing_attribute:{requested_attribute}", [entity], mode,
            )

        if self._is_discovery_query(query):
            print("[Guardrail] status=PASS query_type='recommendation'", flush=True)
        elif subjects["known"]:
            print(f"[Guardrail] status=PASS entities={subjects['known']!r}", flush=True)

        route_plan = None
        distance_plan = None
        if self._is_route_question(query):
            route_places = self._extract_place_entities(retrieval_query)
            if len(route_places) >= 2:
                route_plan = self._build_route_plan(route_places, chunks, query=query)
        elif self._is_distance_question(query):
            distance_places = self._extract_place_entities(retrieval_query)
            if len(distance_places) >= 2:
                place_a, place_b = distance_places[:2]
                distance = self.resolve_pairwise_distance(place_a, place_b, chunks)
                if distance["meters"] is None:
                    answer = (
                        f"ยังไม่มีหลักฐานหรือพิกัดเพียงพอสำหรับยืนยันระยะห่างระหว่าง"
                        f"{place_a}กับ{place_b}ครับ"
                    )
                else:
                    answer = f"{place_a}กับ{place_b}มี{distance['description']}ครับ"
                distance_plan = {
                    "allowed_places": distance_places[:2],
                    "distance": distance,
                    "answer": answer,
                }
        
        # 2. Build structured text + graph context for the selected LLM.
        context_str = self.build_context(chunks)

        # 3. Model Routing
        if target_llm in ["groq", "api", "cloud"]:
            provider, model_name = "groq", self.groq_model
            route_reason = "explicit cloud provider override"
        elif target_llm in ["ollama", "local"]:
            provider, model_name = "ollama", self.local_model
            route_reason = "explicit local provider override"
        else:
            provider, model_name = self.route_model(query, chunks)
            route_reason = self._last_route_reason

        self._log_route(query, provider, model_name, route_reason)

        # 4. Construct Prompt Messages with Session History
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        user_parts = [
            f"ข้อมูลบริบทอ้างอิง:\n{context_str}\n\n",
            f"คำถามของนักท่องเที่ยว: {query}\n",
        ]
        if retrieval_query != query.strip():
            user_parts.append(
                f"คำถามที่แก้คำอ้างอิงจากบทสนทนาแล้ว: {retrieval_query}\n"
            )
        user_parts.append(
            "ข้อกำชับ: ตอบจากหลักฐานบริบทด้านบนเท่านั้น "
            "และรักษาตัวเลขกับเงื่อนไขของผู้ใช้ให้ตรงทุกข้อ\n"
        )
        if self._is_route_question(query):
            user_parts.append(
                "สำหรับคำถามเส้นทาง: ตอบสั้นเป็นรายการลำดับ 1, 2, 3 โดยใช้เฉพาะ"
                "ความสัมพันธ์หรือระยะทางที่มีในหลักฐาน ห้ามกล่าวว่าสถานที่อยู่ใกล้ตัวเอง "
                "ห้ามแต่งระยะทางหรือเหตุผล และหากหลักฐานไม่พอให้แจ้งตรง ๆ\n"
            )
        if route_plan:
            user_parts.append(
                "แผนเส้นทางที่คำนวณจากข้อมูลโครงสร้างแล้ว (ห้ามเพิ่มสถานที่หรือเปลี่ยนลำดับ):\n"
                f"{route_plan['answer']}\n"
            )
        if distance_plan:
            user_parts.append(
                "ผลระยะทางที่ตรวจจากข้อมูลโครงสร้างแล้ว (ห้ามเปลี่ยนค่า/ประเภทระยะทาง):\n"
                f"{distance_plan['answer']}\n"
            )
        user_parts.append("คำตอบของน้องสิงขร:")
        user_content = "".join(user_parts)
        messages.append({"role": "user", "content": user_content})

        # 5. Call Selected LLM with Automatic Failover
        if provider == "groq" and self.groq_api_key:
            answer = self.call_groq(messages, model_name)
            if (answer.startswith("Groq Error") or answer.startswith("Cloud API Error")):
                # Automatic failover to local Ollama
                fallback_reason = (
                    "cloud HTTP error" if answer.startswith("Groq Error")
                    else "cloud connection/runtime error"
                )
                fallback_ans = self.call_ollama(messages, self.local_model)
                fallback_succeeded = not fallback_ans.startswith((
                    "Ollama Error", "Local LLM Error"
                ))
                self._log_fallback(
                    "groq", "ollama", fallback_reason, fallback_succeeded
                )
                if fallback_succeeded:
                    answer = fallback_ans
                    provider = "ollama (failover)"
                    model_name = self.local_model
        else:
            answer = self.call_ollama(messages, model_name)
            if (answer.startswith("Ollama Error") or answer.startswith("Local LLM Error")) and self.groq_api_key:
                # Automatic failover to Groq API
                fallback_reason = (
                    "local HTTP error" if answer.startswith("Ollama Error")
                    else "local connection/runtime error"
                )
                fallback_ans = self.call_groq(messages, self.groq_model)
                fallback_succeeded = not fallback_ans.startswith((
                    "Groq Error", "Cloud API Error"
                ))
                self._log_fallback(
                    "ollama", "groq", fallback_reason, fallback_succeeded
                )
                if fallback_succeeded:
                    answer = fallback_ans
                    provider = "groq (failover)"
                    model_name = self.groq_model

        # Clean answer: remove all markdown formatting noise, greetings, and stray Chinese characters
        clean_ans = answer.strip()
        clean_ans = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_ans)
        clean_ans = re.sub(r'\*(.*?)\*', r'\1', clean_ans)
        clean_ans = re.sub(r'#+\s*', '', clean_ans)
        clean_ans = re.sub(r'^\s*สวัสดี.*?(ค่ะ|ครับ)[!🏮\s]*\n*', '', clean_ans)
        clean_ans = clean_ans.replace("墙壁", "กำแพง")
        clean_ans = re.sub(r'[\u4e00-\u9fff]+', '', clean_ans)
        answer = clean_ans.strip()
        if self._is_route_question(query) and self._has_self_relation_claim(answer):
            answer = "ข้อมูลหลักฐานไม่เพียงพอที่จะยืนยันลำดับเส้นทางครับ"
        if route_plan:
            answer_places = self._extract_place_entities(answer)
            allowed_places = route_plan["allowed_places"]
            ordered_places = route_plan["ordered_places"]
            has_unrelated_place = any(place not in allowed_places for place in answer_places)
            misses_place = any(place not in answer_places for place in allowed_places)
            positions = [answer.find(place) for place in ordered_places]
            wrong_order = any(
                current > following
                for current, following in zip(positions, positions[1:])
                if current >= 0 and following >= 0
            )
            known_descriptions = [
                leg["distance"]["description"]
                for leg in route_plan["legs"]
                if leg["distance"]["meters"] is not None
            ]
            misses_distance = any(
                description not in answer for description in known_descriptions
            )
            if has_unrelated_place or misses_place or wrong_order or misses_distance:
                answer = route_plan["answer"]
            # Route text is already fully grounded and tourist-friendly. Always
            # use it verbatim so generation cannot append unsupported notes.
            answer = route_plan["answer"]
        if distance_plan:
            answer_places = self._extract_place_entities(answer)
            allowed_places = distance_plan["allowed_places"]
            has_unrelated_place = any(place not in allowed_places for place in answer_places)
            misses_place = any(place not in answer_places for place in allowed_places)
            description = distance_plan["distance"]["description"]
            misses_distance = (
                distance_plan["distance"]["meters"] is not None
                and description not in answer
            )
            if has_unrelated_place or misses_place or misses_distance:
                answer = distance_plan["answer"]
        if any(marker in answer for marker in (
            "ยังไม่พบข้อมูล", "ไม่มีข้อมูล", "ข้อมูลไม่เพียงพอ",
            "ไม่สามารถยืนยัน", "หลักฐานไม่เพียงพอ", "ไม่ได้ระบุ",
        )):
            fact_answer = None
            if len(subjects["known"]) == 1 and requested_attribute:
                fact_answer = self._fact_attribute_answer(
                    subjects["known"][0], requested_attribute
                )
            answer = fact_answer or self._collapse_unsupported_answer(answer)

        latency = time.time() - start_time

        # 6. Update Chat Session
        if user_id:
            self.sessions.setdefault(user_id, []).extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer}
            ])
            # Keep sliding window max 6 messages
            if len(self.sessions[user_id]) > 6:
                self.sessions[user_id] = self.sessions[user_id][-6:]

        return {
            "query": query,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "provider": provider,
            "model": model_name,
            "latency_seconds": round(latency, 2),
            "chunks_count": len(chunks),
            "sources": [c.get("title") for c in chunks],
            "mode": mode,
            "allowed_places": (
                route_plan["allowed_places"] if route_plan
                else distance_plan["allowed_places"] if distance_plan else []
            ),
            "route_plan": route_plan,
            "distance_plan": distance_plan,
            "guardrail_status": "PASS",
            "guardrail_reason": "supported_or_discovery",
            "resolved_entities": (
                subjects["known"] or self._extract_place_entities(retrieval_query)
            ),
        }
