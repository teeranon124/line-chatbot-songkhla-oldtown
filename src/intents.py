# -*- coding: utf-8 -*-
"""
Intent Classification Module for Songkhla Old Town Assistant.
Uses Dual-Engine Architecture:
1. Lexical Pattern & Keyword Matching (Deterministic & Instant)
2. Semantic Similarity Fallback via SentenceTransformer Embeddings
"""
import re
from typing import Dict, Any


class SongkhlaIntentClassifier:
    """Classifies user queries into tourist assistance intents."""

    INTENT_KEYWORDS = {
        "home": [
            "หน้าแรก", "เริ่มต้น", "เริ่มใหม่", "ช่วยเหลือ", "home", "help", 
            "ตัวช่วยนำทาง", "คู่มือ", "สวัสดี", "หวัดดี", "hello", "hi"
        ],
        "itinerary": [
            "ขอแผนเที่ยว 2 วัน 1 คืน", "ดูแผนเที่ยว 2 วัน 1 คืน", "แผน 2 วัน 1 คืน", "ตารางเที่ยว 2 วัน"
        ],
        "food": [
            "สำรวจร้านอาหารทั้งหมด", "ดูร้านอาหารทั้งหมด", "ร้านอาหารทั้งหมด", "รวมร้านอาหาร", "แกลเลอรีร้านอาหาร"
        ],
        "attractions": [
            "สำรวจที่เที่ยวทั้งหมด", "ดูที่เที่ยวทั้งหมด", "จุดเช็คอินทั้งหมด", "แกลเลอรีที่เที่ยว", "รวมที่เที่ยว"
        ],
        "hotels": [
            "สำรวจที่พักทั้งหมด", "ดูที่พักทั้งหมด", "โรงแรมทั้งหมด", "รวมที่พัก"
        ],
        "emergency": [
            "ขอเบอร์ฉุกเฉิน", "ฉุกเฉิน", "เบอร์ฉุกเฉิน", "สายด่วนฉุกเฉิน", "ตำรวจท่องเที่ยว", "แจ้งเหตุด่วน"
        ],
        "reset": [
            "รีเซ็ต", "reset", "เริ่มใหม่", "ล้างประวัติ", "clear"
        ]
    }

    INTENT_PROTOTYPES = {
        "home": "สวัสดีครับ ขอหน้าแรกและตัวช่วยนำทางระบบ",
        "itinerary": "ดูแผนเที่ยว 2 วัน 1 คืน",
        "food": "สำรวจร้านอาหารและของหวานทั้งหมด",
        "attractions": "สำรวจสถานที่ท่องเที่ยวทั้งหมด",
        "hotels": "สำรวจโรงแรมและที่พักทั้งหมด",
        "emergency": "ขอเบอร์โทรฉุกเฉินและสายด่วน"
    }

    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self.prototype_embeddings = None
        
        if self.embedding_model:
            try:
                keys = list(self.INTENT_PROTOTYPES.keys())
                texts = [self.INTENT_PROTOTYPES[k] for k in keys]
                self.keys = keys
                self.prototype_embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
            except Exception as e:
                print(f"[Intent Warning] Could not precompute embeddings: {e}")

    def normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r'[\s\.\,\!\?]+', ' ', text)
        return text

    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classifies incoming query string.
        Returns: {"intent": str, "method": str, "confidence": float}
        """
        clean_text = self.normalize(query)

        # 1. Exact / Short Reset check
        if clean_text in ["รีเซ็ต", "reset", "clear", "ล้างแชท"]:
            return {"intent": "reset", "method": "exact", "confidence": 1.0}

        # 2. Emergency Check (Must take priority to provide direct call buttons)
        emergency_keywords = ["ฉุกเฉิน", "สายด่วน", "ตำรวจท่องเที่ยว", "โรงพยาบาล", "กู้ภัย", "แจ้งความ"]
        if any(ek in clean_text for ek in emergency_keywords):
            return {"intent": "emergency", "method": "emergency_rule", "confidence": 1.0}

        # 3. Interrogative & Specific Fact Check -> Direct to Hybrid RAG
        question_markers = [
            "กี่โมง", "เปิดกี่โมง", "ปิดกี่โมง", "เปิดวันไหน", "ปิดวันไหน", "เวลาทำการ",
            "ราคาเท่าไหร่", "กี่บาท", "ราคา", "เบอร์", "โทร", "โทรศัพท์",
            "อยู่ที่ไหน", "อยู่ตรงไหน", "พิกัด", "ทางไป", "ไปยังไง", "เดินทางยังไง",
            "ประวัติ", "ความเป็นมา", "คืออะไร", "ทำไม", "อย่างไร", "สร้างเมื่อ",
            "ยุคไหน", "ขายอะไร", "เมนูอะไรเด็ด", "จอดรถ", "ค่าเข้า", "ค่าตั๋ว"
        ]
        if any(marker in clean_text for marker in question_markers):
            return {"intent": "qa_hybrid", "method": "question_marker", "confidence": 0.98}

        # 3. Pattern & Keyword Matching for Broad Browsing
        for intent, kw_list in self.INTENT_KEYWORDS.items():
            for kw in kw_list:
                if kw in clean_text:
                    if len(clean_text) <= 35:
                        return {"intent": intent, "method": "keyword_rule", "confidence": 0.95}

        # 3. Semantic Embedding Fallback (if query is relatively short and matches a prototype)
        if self.embedding_model and self.prototype_embeddings is not None and len(clean_text) <= 40:
            try:
                q_emb = self.embedding_model.encode([clean_text], normalize_embeddings=True)
                import numpy as np
                sims = np.dot(self.prototype_embeddings, q_emb.T).flatten()
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                if best_score > 0.65:
                    return {
                        "intent": self.keys[best_idx],
                        "method": "semantic_embedding",
                        "confidence": round(best_score, 3)
                    }
            except Exception:
                pass

        # 4. Default to Hybrid RAG (Specific Question / Multi-hop / Custom Planning)
        return {"intent": "qa_hybrid", "method": "default_rag", "confidence": 1.0}
