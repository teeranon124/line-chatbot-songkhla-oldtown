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
6. อ้างอิงข้อมูลจากบริบทอย่างเคร่งครัด หากไม่มีข้อมูลให้ตอบตามตรงว่าไม่มีข้อมูล ห้ามกุเรื่องขึ้นมาเอง"""


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
        
        # 1. Retrieve Context
        chunks = self.retriever.retrieve(query, top_k=top_k, mode=mode)
        
        # 2. Build structured text + graph context for the selected LLM.
        context_str = self.build_context(chunks)

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
                # Automatic failover to local Ollama
                fallback_ans = self.call_ollama(messages, self.local_model)
                if not (fallback_ans.startswith("Ollama Error") or fallback_ans.startswith("Local LLM Error")):
                    answer = fallback_ans
                    provider = "ollama (failover)"
                    model_name = self.local_model
        else:
            answer = self.call_ollama(messages, model_name)
            if (answer.startswith("Ollama Error") or answer.startswith("Local LLM Error")) and self.groq_api_key:
                # Automatic failover to Groq API
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
