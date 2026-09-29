# -*- coding: utf-8 -*-
"""
RAG Engine Orchestrator for Songkhla Old Town Assistant.
Features:
- Dual-Engine LLM Generation (Local Ollama vs Cloud Groq API)
- Intelligent Model Router (Factual -> Local, Complex Reasoning -> Cloud API)
- Sliding-window Multi-turn Chat Memory
- Guardrails & Strict Grounding in AnyFlip + Scraped Facts
"""
import re
import time
import requests
from typing import List, Dict, Any, Tuple

from .config import models, paths
from .retriever import HybridRetriever
from .graph_engine import SongkhlaGraphEngine


SYSTEM_PROMPT = """คุณคือ "น้องสิงขร" ผู้ช่วยอัจฉริยะนำเที่ยวย่านเมืองเก่าสงขลา
หน้าที่ของคุณ:
1. ตอบให้ 'สั้น กระชับ ตรงประเด็น' ไม่เกิน 3-4 บรรทัด
2. ห้ามมีคำเกริ่นทักทายเยิ่นเย้อ เช่น "สวัสดีค่ะ ยินดีต้อนรับ..." หรือ "น้องสิงขรขอแนะนำ..." ให้ตอบเข้าเนื้อหาทันที
3. ห้ามใช้เครื่องหมาย Markdown เช่น เครื่องหมายดอกจัน ** หรือเครื่องหมาย # เด็ดขาด ให้ใช้ภาษาไทยธรรมดาที่เป็นธรรมชาติ
4. ต้องตอบเป็นภาษาไทยล้วน 100% ห้ามมีตัวอักษรจีนหรือภาษาต่างประเทศปะปนเด็ดขาด (เช่น ห้ามใช้คำว่า 墙壁 ให้ใช้คำว่า กำแพงหรือผนัง)
5. หากถามเรื่องของหวานหรือของกินเล่น ให้เลือกเฉพาะร้านของหวาน เช่น ร้านไอติมโอ่ง หรือบ้านขนมไทยสองแสน ห้ามนำร้านอาหารคาวมาตอบเป็นของหวาน
6. กฎเหล็กป้องกันภาพหลอนรอบด้าน (Universal Anti-Hallucination Guardrails): อ้างอิงข้อมูลจากบริบทอย่างเคร่งครัด 100% ห้ามกุเรื่องขึ้นมาเองเด็ดขาด
   - [เบอร์โทรศัพท์]: ห้ามสุ่มหรือแต่งเบอร์โทรศัพท์เด็ดขาด หากไม่มีให้ตอบว่า "ไม่มีการระบุเบอร์โทรศัพท์ติดต่อ"
   - [เวลาเปิด-ปิด]: ห้ามเดาเวลาเปิดปิดเด็ดขาด หากไม่มีตัวเลขเวลาในบริบท ให้ตอบว่า "ไม่มีการระบุเวลาเปิด-ปิดที่แน่นอน"
   - [ราคา/ค่าเข้าชม]: ห้ามกุตัวเลขราคาเองเด็ดขาด หากเข้าชมฟรีให้ระบุว่าฟรี หากไม่มีราคาให้ตอบว่า "ไม่มีข้อมูลราคา"
   - [ตำแหน่งถนน]: ต้องระบุชื่อถนนให้ตรงตามบริบท ห้ามสลับหรือเดาชื่อถนนเด็ดขาด"""


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

    def route_model(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Adaptive Model Router based on Information Entropy, Context Density & Multi-hop Breadth.
        Routes to Cloud API (Groq) for high-entropy multi-domain synthesis,
        and Local LLM (Ollama) for low-entropy factual answering.
        Does NOT rely on brittle keyword matching.
        """
        if not self.groq_api_key:
            return "ollama", self.local_model

        # 1. Context Information Volume (total length of retrieved evidence)
        total_context_len = sum(len(c.get("content", "")) for c in context_chunks)
        
        # 2. Multi-Domain Entity Diversity (categories spanning different aspects)
        categories = {c.get("category", "") for c in context_chunks if c.get("category")}
        
        # 3. Query Structural Complexity and Retrieval Spread
        q_len = len(query.strip())
        num_chunks = len(context_chunks)

        # Composite Complexity Scoring:
        # - Broad multi-chunk spread (>= 5 chunks retrieved by adaptive retriever)
        # - High context volume (> 1,200 chars) requires high-parameter reasoning
        # - Multi-category synthesis (>= 3 distinct domain categories in context)
        is_high_complexity = (
            (num_chunks >= 5 and total_context_len > 1200) or
            (len(categories) >= 3) or
            (q_len > 40 and num_chunks >= 4)
        )

        if is_high_complexity:
            return "groq", self.groq_model
        return "ollama", self.local_model

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

    def verify_and_ground_answer(self, query: str, answer: str, context_str: str) -> str:
        """
        Universal Multi-Dimensional Anti-Hallucination Guardrail:
        1. Phone Number Verification (Regex cross-check against context, supports (074), +66, and mobile formats)
        2. Operating Hours & Times Verification (Strict time digit grounding against context; no false positive pass-through)
        3. Pricing & Fees Verification (Price digits against context or free admission indicator)
        4. Location & Street Consistency Validation (Entity-specific Graph Topology grounding; no cross-entity corruption)
        """
        clean_context = context_str.replace(" ", "")

        # 1. Phone Numbers Guardrail: Supports (074), +66, and standard formats
        clean_ctx_digits = re.sub(r'\D', '', clean_context)
        phone_pattern = r'(?:\+66[-\s]?\(?\d{1,2}\)?|\(?0\d{1,2}\)?)[-\s]?\d{3,4}[-\s]?\d{3,4}\b'
        phone_matches = re.findall(phone_pattern, answer)
        if phone_matches:
            for phone in phone_matches:
                clean_phone_digits = re.sub(r'\D', '', phone)
                if clean_phone_digits.startswith("66"):
                    clean_phone_digits = "0" + clean_phone_digits[2:]
                
                # Verify if phone digits exist in context
                if clean_phone_digits not in clean_ctx_digits and (clean_phone_digits[1:] not in clean_ctx_digits):
                    if any(k in query for k in ["เบอร์", "โทร", "ติดต่อ"]):
                        return "ขออภัยครับ ในฐานข้อมูลยังไม่มีการระบุเบอร์โทรศัพท์ติดต่อของสถานที่ดังกล่าว"
                    answer = re.sub(re.escape(phone), "[ไม่มีข้อมูลเบอร์]", answer)

        # 2. Operating Hours & Times Guardrail: Strict digit & format verification
        if any(k in query for k in ["เวลา", "กี่โมง", "เปิดกี่โมง", "ปิดกี่โมง", "เปิดปิด", "เวลาเปิด", "เปิดทำ"]):
            time_matches = re.findall(r'\b\d{1,2}[:.]\d{2}(?:\s*(?:น\.|น|โมง))?\b', answer)
            if time_matches:
                ctx_normalized = clean_context.replace(".", ":")
                time_in_ctx = False
                for tm in time_matches:
                    clean_tm = re.sub(r'[^\d:]', '', tm.replace(".", ":"))
                    if clean_tm and clean_tm in ctx_normalized:
                        time_in_ctx = True
                        break
                    tm_digits = re.sub(r'\D', '', tm)
                    if len(tm_digits) >= 3 and tm_digits in re.sub(r'\D', '', clean_context):
                        time_in_ctx = True
                        break
                if not time_in_ctx:
                    return "ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุเวลาเปิด-ปิดที่แน่นอน แนะนำตรวจสอบกับทางสถานที่โดยตรงครับ"

        # 3. Pricing & Admission Fees Guardrail: Strict price grounding
        if any(k in query for k in ["ราคา", "ค่าเข้า", "กี่บาท", "เท่าไหร่", "ค่าบัตร"]):
            price_matches = re.findall(r'\b\d{1,5}\s*(?:บาท|฿)\b', answer)
            if price_matches:
                ctx_digits = re.sub(r'\D', '', clean_context)
                has_grounded_price = False
                for pm in price_matches:
                    pm_digits = re.sub(r'\D', '', pm)
                    if pm_digits and pm_digits in ctx_digits:
                        has_grounded_price = True
                        break
                if not has_grounded_price:
                    if "ฟรี" in clean_context:
                        return "สถานที่นี้เปิดให้เข้าชมฟรี ไม่มีค่าใช้จ่ายครับ"
                    return "ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุราคาหรือค่าเข้าชมที่แน่ชัด แนะนำสอบถามหน้าร้านครับ"

        # 4. Street / Location Knowledge Graph Ground-Truth Validation (Entity-Specific)
        known_locations = {
            "แต้เฮี้ยงอิ๋ว": "ถนนนางงาม",
            "เกียดฟั่ง": "ถนนนางงาม",
            "ไอติมโอ่ง": "ถนนนางงาม",
            "ศาลเจ้าพ่อหลักเมือง": "ถนนนางงาม",
            "บ้านขนมไทยสองแสน": "ถนนนางงาม",
            "หับ โห้ หิ้น": "ถนนนครนอก",
            "โรงสีแดง": "ถนนนครนอก",
            "บ้านนครใน": "ถนนนครนอก",
            "มัสยิดบ้านบน": "ถนนพัทลุง",
            "กำแพงเมืองสงขลา": "ถนนจะนะ"
        }
        streets = ["ถนนนางงาม", "ถนนนครนอก", "ถนนนครใน", "ถนนพัทลุง", "ถนนจะนะ"]
        all_entities = list(known_locations.keys())
        
        for entity, correct_street in known_locations.items():
            if entity in answer:
                for s in streets:
                    if s != correct_street:
                        # Negative lookahead ensures we only match within the entity's clause
                        # and never bridge across another known entity or major sentence delimiter
                        other_ents = '|'.join(re.escape(e) for e in all_entities if e != entity)
                        stop_pattern = rf'(?:[。\.\n;]|ส่วน|และ|ขณะที่|{other_ents})'
                        
                        p1 = rf'({re.escape(entity)}(?:(?!{stop_pattern})[\s\S]){{0,40}}?)({re.escape(s)})'
                        answer = re.sub(p1, rf'\g<1>{correct_street}', answer)
                        
                        p2 = rf'({re.escape(s)}(?:(?!{stop_pattern})[\s\S]){{0,40}}?)({re.escape(entity)})'
                        answer = re.sub(p2, rf'{correct_street}\g<2>', answer)

        return answer

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
        
        # 1. Retrieve Context
        chunks = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        
        # 2. Build Context String
        context_parts = []
        for i, c in enumerate(chunks, 1):
            title = c.get("title", f"เอกสารที่ {i}")
            content = c.get("content", "").strip()
            context_parts.append(f"--- [เอกสารที่ {i}: {title}] ---\n{content}\n")
        context_str = "\n".join(context_parts)

        # 3. Model Routing
        if target_llm in ["groq", "api", "cloud"]:
            provider, model_name = "groq", self.groq_model
        elif target_llm in ["ollama", "local"]:
            provider, model_name = "ollama", self.local_model
        else:
            provider, model_name = self.route_model(query, chunks)

        # 4. Construct Prompt Messages with Session History
        history = self.sessions.get(user_id, [])[-4:]  # Last 2 turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        user_content = (
            f"ข้อมูลบริบทอ้างอิง:\n{context_str}\n\n"
            f"คำถามของนักท่องเที่ยว: {query}\n"
            f"คำตอบของน้องสิงขร:"
        )
        messages.append({"role": "user", "content": user_content})

        # 5. Call Selected LLM with Automatic Failover
        if provider == "groq" and self.groq_api_key:
            answer = self.call_groq(messages, model_name)
            if (answer.startswith("Groq Error") or answer.startswith("Cloud API Error")):
                fallback_ans = self.call_ollama(messages, self.local_model)
                if not (fallback_ans.startswith("Ollama Error") or fallback_ans.startswith("Local LLM Error")):
                    answer = fallback_ans
                    provider = "ollama (failover)"
                    model_name = self.local_model
        else:
            answer = self.call_ollama(messages, model_name)
            if (answer.startswith("Ollama Error") or answer.startswith("Local LLM Error")) and self.groq_api_key:
                fallback_ans = self.call_groq(messages, self.groq_model)
                if not (fallback_ans.startswith("Groq Error") or fallback_ans.startswith("Cloud API Error")):
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

        # 5.1 Universal Deterministic Anti-Hallucination Guardrail (Fact Verification)
        clean_ans = self.verify_and_ground_answer(query, clean_ans, context_str)
        answer = clean_ans.strip()

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
            "answer": answer,
            "provider": provider,
            "model": model_name,
            "latency_seconds": round(latency, 2),
            "chunks_count": len(chunks),
            "sources": [c.get("title") for c in chunks],
            "mode": mode
        }
