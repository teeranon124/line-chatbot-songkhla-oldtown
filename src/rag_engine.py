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
6. อ้างอิงข้อมูลจากบริบทอย่างเคร่งครัด หากไม่มีข้อมูลให้ตอบตามตรงว่าไม่มีข้อมูล ห้ามกุเรื่องขึ้นมาเอง
7. เวลา ราคา ระยะทาง และตัวเลขทุกชนิด ต้องคัดตามหลักฐานตรงตัว ห้ามประมาณหรือเปลี่ยนตัวเลข
8. กล่าวถึงเฉพาะสถานที่และความสัมพันธ์ที่ปรากฏในหลักฐาน ห้ามเพิ่มสถานที่หรือเชื่อมโยงข้อมูลเอง
9. คำแนะนำต้องตรงทุกเงื่อนไขที่ผู้ใช้ระบุ เช่น ประเภทอาหาร งบประมาณ และพื้นที่ หากหลักฐานไม่ครบให้แจ้งว่าไม่มีข้อมูลเพียงพอ"""


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
        """Load short canonical place names for resolving one-turn follow-ups."""
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
            "mode": mode
        }
