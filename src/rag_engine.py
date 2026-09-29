# -*- coding: utf-8 -*-
"""
rag_engine.py
Core Hybrid GraphRAG Generation Engine for Songkhla Old Town Assistant.
Features:
- Dual-Track Retrieval: Combines Dense FAISS, Sparse BM25, and NetworkX Graph
- Adaptive Model Router (Ollama Local vs Groq Cloud API)
- Sliding Window Context Management (Multi-turn conversational memory)
- Strict Anti-Hallucination Constraints and Grounding Verification
"""

import re
import time
import json
import requests
from typing import Dict, List, Any, Tuple, Optional

from src.retriever import HybridRetriever
from src.graph_engine import SongkhlaGraphEngine
from src.config import models, retrieval, paths


SYSTEM_PROMPT = """คุณคือ "น้องสิงขร" ผู้ช่วยอัจฉริยะนำเที่ยวย่านเมืองเก่าสงขลา
หน้าที่ของคุณคือให้ข้อมูลแก่นักท่องเที่ยวอย่างเป็นมิตร ถูกต้อง ชัดเจน และกระชับ

กฎการตอบคำถาม:
1. ให้ตอบตรงประเด็น สุภาพ และใช้ภาษาที่เข้าใจง่าย (ความยาวประมาณ 2-4 ประโยค ไม่เยิ่นเย้อ)
2. ห้ามใช้เครื่องหมายดอกจัน (asterisk) เช่น **ข้อความ** หรือ *ข้อความ* ในการเน้นคำเด็ดขาด ให้เขียนเป็นข้อความธรรมดา
3. ห้ามใช้หัวข้อย่อยแบบ Markdown (# หรือ ##) ให้ขึ้นบรรทัดใหม่ธรรมดา
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
        self._known_place_names = None
        self._last_route_reason = "default routing"

    def _get_known_place_names(self) -> List[str]:
        if getattr(self, "_known_place_names", None) is not None:
            return self._known_place_names

        names = set()
        places_path = getattr(paths, "places_facts_path", getattr(paths, "facts_path", None))
        if places_path:
            try:
                with open(places_path, "r", encoding="utf-8") as f:
                    places = json.load(f)
                for p in places:
                    name = p.get("name")
                    if name:
                        names.add(name)
                        names.add(name.split("(", 1)[0].strip())
            except (OSError, ValueError, TypeError):
                pass

        self._known_place_names = sorted(
            (name for name in names if name), key=len, reverse=True
        )
        return self._known_place_names

    def resolve_retrieval_query(
        self, query: str, history: List[Dict[str, str]]
    ) -> str:
        """Attach the immediately previous place to an ambiguous follow-up."""
        clean_query = query.strip()
        place_names = self._get_known_place_names()
        if not history or not place_names:
            return clean_query
        if any(name in clean_query for name in place_names):
            return clean_query

        is_follow_up = bool(re.match(
            r"^(แล้ว(?:ล่ะ|ละ)?|ส่วน|ที่นั่น|ที่นี่)\s*", clean_query
        ))
        if not is_follow_up:
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
        referent = next(
            (name for text in (previous_user, previous_assistant)
             for name in place_names if name in text),
            None
        )
        if not referent:
            return clean_query

        follow_up = re.sub(
            r"^(แล้ว(?:ล่ะ|ละ)?|ส่วน|ที่นั่น|ที่นี่)\s*", "", clean_query
        ).strip()
        return f"{referent} {follow_up}" if follow_up else referent

    def route_model(self, query: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Adaptive Model Router based on Information Entropy, Context Density & Multi-hop Breadth.
        Routes to Cloud API (Groq) for high-entropy multi-domain synthesis,
        and Local LLM (Ollama) for low-entropy factual answering.
        """
        if not self.groq_api_key:
            self._last_route_reason = "Groq is not configured, using local"
            return "ollama", self.local_model

        # 1. Context Information Volume (total length of retrieved evidence)
        total_context_len = sum(len(c.get("content", "")) for c in context_chunks)
        
        # 2. Multi-Domain Entity Diversity (categories spanning different aspects)
        categories = {c.get("category", "") for c in context_chunks if c.get("category")}
        
        # 3. Query Structural Complexity and Retrieval Spread
        q_len = len(query.strip())
        num_chunks = len(context_chunks)

        is_high_complexity = (
            (num_chunks >= 5 and total_context_len > 1200) or
            (len(categories) >= 3) or
            (q_len > 40 and num_chunks >= 4)
        )

        if is_high_complexity:
            self._last_route_reason = "complex planning/synthesis query"
            return "groq", self.groq_model
        
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
        print(f"Original provider: {labels.get(original_provider, original_provider)}", flush=True)
        print(f"Fallback provider: {labels.get(fallback_provider, fallback_provider)}", flush=True)
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

    def verify_and_ground_answer(self, query: str, answer: str, context_str: str) -> str:
        """
        Anti-Hallucination Guardrail:
        1. Phone Number Verification
        2. Operating Hours & Times Verification
        3. Pricing & Fees Verification
        4. Location & Street Consistency Validation
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
            "แต้เฮี้ยงอิ้ว": "ถนนนางงาม",
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
        
        history = self.sessions.get(user_id, [])[-4:]  # Last 2 turns
        retrieval_query = self.resolve_retrieval_query(query, history)

        # 1. Retrieve Context
        chunks = self.retriever.retrieve(retrieval_query, top_k=top_k, mode=mode)
        
        # 2. Build Context String with structured text + graph evidence
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
            route_reason = getattr(self, "_last_route_reason", "adaptive routing")

        self._log_route(query, provider, model_name, route_reason)

        # 4. Construct Prompt Messages with Session History
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

        user_content = (
            f"ข้อมูลบริบทอ้างอิง:\n{context_str}\n\n"
            f"คำถามของนักท่องเที่ยว: {query}\n"
            "ข้อกำชับ: ตอบจากหลักฐานบริบทด้านบนเท่านั้น และรักษาตัวเลขกับเงื่อนไขของผู้ใช้ให้ตรงทุกข้อ\n"
            f"คำตอบของน้องสิงขร:"
        )
        messages.append({"role": "user", "content": user_content})

        # 5. Call Selected LLM with Automatic Failover
        if provider == "groq" and self.groq_api_key:
            answer = self.call_groq(messages, model_name)
            if (answer.startswith("Groq Error") or answer.startswith("Cloud API Error")):
                fallback_reason = "cloud error"
                fallback_ans = self.call_ollama(messages, self.local_model)
                fallback_succeeded = not fallback_ans.startswith(("Ollama Error", "Local LLM Error"))
                self._log_fallback("groq", "ollama", fallback_reason, fallback_succeeded)
                if fallback_succeeded:
                    answer = fallback_ans
                    provider = "ollama (failover)"
                    model_name = self.local_model
        else:
            answer = self.call_ollama(messages, model_name)
            if (answer.startswith("Ollama Error") or answer.startswith("Local LLM Error")) and self.groq_api_key:
                fallback_reason = "local error"
                fallback_ans = self.call_groq(messages, self.groq_model)
                fallback_succeeded = not fallback_ans.startswith(("Groq Error", "Cloud API Error"))
                self._log_fallback("ollama", "groq", fallback_reason, fallback_succeeded)
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
            "mode": mode
        }
