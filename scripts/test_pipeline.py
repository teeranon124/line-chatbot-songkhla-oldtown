# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import models, paths
from src.intents import SongkhlaIntentClassifier
from src.graph_engine import SongkhlaGraphEngine
from src.retriever import HybridRetriever
from src.rag_engine import SongkhlaRAGEngine
from src.line_handler import SongkhlaLineHandler

print("=== ALL MODULES IMPORTED SUCCESSFULLY ===")

engine = SongkhlaRAGEngine()
handler = SongkhlaLineHandler(rag_engine=engine)

# 1. Test Intent Classification
test_queries = [
    "สวัสดีครับ",
    "ขอแผนเที่ยว 2 วัน 1 คืน",
    "แนะนำของกินอร่อย",
    "ที่เที่ยวถ่ายรูป",
    "โรงแรมที่พัก",
    "ขอเบอร์ฉุกเฉิน",
    "ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่"
]

print("\n--- Intent Classification Test ---")
for q in test_queries:
    res = handler.intent_classifier.classify(q)
    print(f"Query: '{q}' -> Intent: {res['intent']} (method: {res['method']}, conf: {res['confidence']})")

# 2. Test RAG Generation with Local Ollama
print("\n--- Testing RAG Generation with Local Ollama ---")
r_local = engine.generate("ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่", target_llm="ollama")
print(f"Ollama [Latency: {r_local['latency_seconds']}s, Model: {r_local['model']}]")
print(f"Answer: {r_local['answer'][:200]}...")

# 3. Test RAG Generation with Cloud Groq API
print("\n--- Testing RAG Generation with Cloud Groq API ---")
r_groq = engine.generate("แนะนำของกินอร่อยบนถนนนางงาม พร้อมเวลาเปิดปิด", target_llm="groq")
print(f"Groq [Latency: {r_groq['latency_seconds']}s, Model: {r_groq['model']}]")
print(f"Answer: {r_groq['answer'][:200]}...")

print("\n=== END-TO-END PIPELINE VERIFICATION PASSED ===")
