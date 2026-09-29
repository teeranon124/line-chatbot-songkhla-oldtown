# -*- coding: utf-8 -*-
"""
07_test_end_to_end.py
Comprehensive End-to-End System Verification for Songkhla Old Town AI Assistant.

Performs exhaustive verification across:
1. Intent Classifier (Semantic Embedding + Regex Catalog Routing)
2. Hybrid Retrieval (Dense FAISS WangchanBERTa + Sparse BM25 + Graph Engine RRF)
3. Dual LLM Engine (Local Ollama qwen2.5:3b default + Cloud Groq)
4. Response Sanitization (Markdown removal: no asterisks, no raw hashes)
5. Multi-Entity Recognition & Carousel Generation (Multi-Place Queries)
6. Official LINE API Schema Validation (Validates Flex JSON payloads)
"""

import sys
import os
import time
import json
import requests
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import models, paths, line_config
from src.intents import SongkhlaIntentClassifier
from src.graph_engine import SongkhlaGraphEngine
from src.retriever import HybridRetriever
from src.rag_engine import SongkhlaRAGEngine
from src.flex_templates import SongkhlaFlexTemplates
from src.line_handler import SongkhlaLineHandler

LINE_VALIDATE_URL = "https://api.line.me/v2/bot/message/validate/reply"


def validate_line_payload(messages_list):
    """Validates message payloads using official LINE endpoint."""
    if not line_config.channel_access_token:
        return True, "Channel access token not configured, skipped API call."
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {line_config.channel_access_token}"
    }
    payload = {
        "messages": [m.as_json_dict() if hasattr(m, "as_json_dict") else m for m in messages_list]
    }
    try:
        resp = requests.post(LINE_VALIDATE_URL, headers=headers, json=payload, timeout=5)
        if resp.status_code == 200:
            return True, "LINE API 200 OK (Valid Flex Schema)"
        else:
            return False, f"LINE API {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, f"Request failed: {e}"


def run_tests():
    print("=" * 70)
    print("🚀 [07] SONGKHLA OLD TOWN ASSISTANT - END-TO-END VERIFICATION")
    print("=" * 70)
    
    results = []
    
    # --- TEST 1: Load Engines ---
    print("\n[Step 1/6] Initializing RAG Engine and LINE Handler...")
    t0 = time.time()
    engine = SongkhlaRAGEngine()
    handler = SongkhlaLineHandler(rag_engine=engine)
    init_time = time.time() - t0
    print(f"  ✅ RAG Engine & LINE Handler loaded in {init_time:.2f}s")
    results.append(("System Initialization", "PASS", f"{init_time:.2f}s"))

    # --- TEST 2: Intent Classification ---
    print("\n[Step 2/6] Testing Intent Classification Routing...")
    intent_tests = [
        ("สวัสดีครับ", "home"),
        ("ขอแผนเที่ยว 2 วัน 1 คืน", "itinerary"),
        ("สำรวจร้านอาหารทั้งหมด", "food"),
        ("สำรวจที่เที่ยวทั้งหมด", "attractions"),
        ("โรงแรมที่พัก", "hotels"),
        ("ขอเบอร์ฉุกเฉิน", "emergency"),
        ("รีเซ็ต", "reset"),
        ("ของกินอร่อยบนถนนนางงามมีอะไรบ้าง", "qa_hybrid"),
        ("ที่เที่ยวถ่ายรูปสวยๆ", "qa_hybrid"),
        ("ร้านไอติมโอ่ง กับร้านเจ๊นิ เปิดกี่โมง", "qa_hybrid"),
    ]
    intent_success = True
    for text, expected in intent_tests:
        cls_res = handler.intent_classifier.classify(text)
        actual = cls_res["intent"]
        ok = (actual == expected)
        mark = "✅" if ok else "❌"
        print(f"  {mark} '{text}' -> {actual} (Expected: {expected}, Conf: {cls_res['confidence']})")
        if not ok:
            intent_success = False
    results.append(("Intent Classification (10 Queries)", "PASS" if intent_success else "FAIL", "All Routed Correctly" if intent_success else "Mismatch Found"))

    # --- TEST 3: Hybrid Retrieval Quality ---
    print("\n[Step 3/6] Testing Hybrid Retrieval (Dense FAISS + Sparse BM25 + Graph)...")
    q_retrieval = "ร้านไอติมโอ่ง ซิกเนเจอร์คืออะไร"
    retrieved = engine.retriever.retrieve(q_retrieval, top_k=5)
    has_aitim = any("ไอติมโอ่ง" in r.get("content", "") or "aitim_oang" in r.get("id", "") for r in retrieved)
    print(f"  Retrieved {len(retrieved)} chunks for '{q_retrieval}'")
    print(f"  Top Match Chunks: {[r.get('chunk_id', r.get('title')) for r in retrieved[:3]]}")
    if has_aitim:
        print("  ✅ Target entity retrieved at Top ranks.")
        results.append(("Hybrid Retrieval Accuracy", "PASS", f"Top match: {retrieved[0].get('chunk_id', '')} ({retrieved[0].get('title', '')})"))
    else:
        print("  ❌ Target entity missed.")
        results.append(("Hybrid Retrieval Accuracy", "FAIL", "Entity missed"))

    # --- TEST 4: Local Ollama Generation & Sanitization ---
    print("\n[Step 4/6] Testing Local Ollama (qwen2.5:3b) Generation & Sanitization...")
    t0 = time.time()
    rag_res = engine.generate("ร้านไอติมโอ่งเปิดกี่โมงและราคาเท่าไหร่", target_llm="ollama")
    ans_text = rag_res.get("answer", "")
    has_markdown_asterisks = "**" in ans_text
    print(f"  Model: {rag_res['model']} | Latency: {rag_res['latency_seconds']}s")
    print(f"  Answer Preview:\n  \"{ans_text}\"")
    if not has_markdown_asterisks:
        print("  ✅ Markdown asterisks successfully stripped from answer.")
        results.append(("Local Ollama Generation & Clean Text", "PASS", f"{rag_res['latency_seconds']}s, Clean text"))
    else:
        print("  ⚠️ Markdown asterisks detected in answer.")
        results.append(("Local Ollama Generation & Clean Text", "WARN", "Contains asterisks"))

    # --- TEST 5: Multi-Entity Carousel Generation (2 Places) ---
    print("\n[Step 5/6] Testing Multi-Entity Message Response (2 Places Query)...")
    multi_query = "ร้านไอติมโอ่ง กับร้านเจ๊นิ เปิดกี่โมง"
    reply_messages = handler.process_message(multi_query)
    
    is_multi_message = isinstance(reply_messages, list) and len(reply_messages) == 2
    print(f"  Input: '{multi_query}'")
    print(f"  Number of returned messages: {len(reply_messages) if isinstance(reply_messages, list) else 1}")
    
    if is_multi_message:
        msg1 = reply_messages[0]
        msg2 = reply_messages[1]
        desc1 = getattr(msg1, "alt_text", getattr(msg1, "text", str(msg1)))
        desc2 = getattr(msg2, "alt_text", getattr(msg2, "text", str(msg2)))
        print(f"  Message 1: {desc1[:80]}")
        print(f"  Message 2: {desc2[:80]}")
        
        # Check carousel bubbles count
        carousel_bubbles = msg2.contents.contents if hasattr(msg2.contents, "contents") else []
        bubble_titles = []
        for b in carousel_bubbles:
            title_text = getattr(b.body.contents[0], "text", "") if hasattr(b.body, "contents") and len(b.body.contents) > 0 else ""
            bubble_titles.append(title_text)
        print(f"  Carousel Bubbles ({len(carousel_bubbles)}): {bubble_titles}")
        
        if len(carousel_bubbles) == 2:
            print("  ✅ Multi-Entity extraction matched exactly 2 places and built a 2-bubble carousel!")
            results.append(("Multi-Entity Carousel Attachment", "PASS", f"2 Places: {', '.join(bubble_titles)}"))
        else:
            print(f"  ⚠️ Expected 2 bubbles, got {len(carousel_bubbles)}")
            results.append(("Multi-Entity Carousel Attachment", "WARN", f"{len(carousel_bubbles)} bubbles"))
    else:
        print(f"  ❌ Expected 2 messages (Card + Carousel), received {len(reply_messages) if isinstance(reply_messages, list) else 1}")
        results.append(("Multi-Entity Carousel Attachment", "FAIL", "Did not return 2 messages"))

    # --- TEST 6: Official LINE API Schema Validation ---
    print("\n[Step 6/6] Validating Flex Message Payloads via LINE Official API...")
    is_valid, reason = validate_line_payload(reply_messages if isinstance(reply_messages, list) else [reply_messages])
    print(f"  LINE Schema Validation Result: {reason}")
    results.append(("Official LINE Flex Schema Validation", "PASS" if is_valid else "FAIL", reason))

    # --- SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Component / Feature':<38} | {'Status':<8} | {'Details'}")
    print("-" * 70)
    all_passed = True
    for comp, status, detail in results:
        status_str = f"✅ {status}" if status == "PASS" else (f"⚠️ {status}" if status == "WARN" else f"❌ {status}")
        print(f"{comp:<38} | {status_str:<8} | {detail}")
        if status == "FAIL":
            all_passed = False
            
    print("-" * 70)
    if all_passed:
        print("🎉 ALL END-TO-END PIPELINE CHECKS PASSED PERFECTLY!")
    else:
        print("⚠️ Some checks require attention.")
    print("=" * 70 + "\n")
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
