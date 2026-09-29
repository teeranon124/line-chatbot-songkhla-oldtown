# -*- coding: utf-8 -*-
"""
LINE Bot Webhook Handler Module for Songkhla Old Town Assistant.
Dispatches incoming messages through Intent Classifier to Rich Flex Carousels or Hybrid RAG.
Routes concise text replies and optional LINE Flex place cards by response type.
"""
import json
from typing import Union, List
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    ImageSendMessage,
    FlexSendMessage
)

from .config import line_config, paths
from .intents import SongkhlaIntentClassifier
from .rag_engine import SongkhlaRAGEngine
from .flex_templates import SongkhlaFlexTemplates


class SongkhlaLineHandler:
    LINE_TEXT_LIMIT = 5000

    def __init__(self, rag_engine: SongkhlaRAGEngine = None):
        self.rag_engine = rag_engine or SongkhlaRAGEngine()
        self.channel_secret = line_config.channel_secret
        self.channel_access_token = line_config.channel_access_token

        # Load places facts for instant carousel population
        with open(paths.facts_path, "r", encoding="utf-8") as f:
            self.places = json.load(f)

        # Separate categories
        self.food_places = [p for p in self.places if "อาหาร" in p["category"] or "ของหวาน" in p["category"] or "คาเฟ่" in p["category"]]
        self.attraction_places = [p for p in self.places if "ประวัติศาสตร์" in p["category"] or "วัฒนธรรม" in p["category"] or "จุดชมวิว" in p["category"] or "ศิลปะ" in p["category"] or "กิจกรรม" in p["category"]]
        self.hotel_places = [p for p in self.places if "โรงแรม" in p["category"] or "ที่พัก" in p["category"]]

        # Initialize Intent Classifier with Dense SBERT Model
        dense_model = getattr(self.rag_engine.retriever.dense, "model", None)
        self.intent_classifier = SongkhlaIntentClassifier(embedding_model=dense_model)

        # Initialize LINE SDK
        if self.channel_access_token and self.channel_secret:
            self.line_bot_api = LineBotApi(self.channel_access_token)
            self.handler = WebhookHandler(self.channel_secret)
            self._register_events()
            print("[LINE Handler] LINE Bot API and Webhook Handler initialized successfully.")
        else:
            self.line_bot_api = None
            self.handler = None
            print("[LINE Handler Notice] LINE credentials not fully configured in .env. Running in Mock/REST API mode.")

    def _register_events(self):
        @self.handler.add(MessageEvent, message=TextMessage)
        def handle_text_message(event):
            user_text = event.message.text.strip()
            user_id = event.source.user_id if hasattr(event.source, "user_id") else "default_user"
            reply_token = event.reply_token

            print(f"\n📨 [Incoming Message] UserID: {user_id} | Text: '{user_text}' | Token: {reply_token[:10]}...", flush=True)

            if reply_token == "dummy_reply_token" or reply_token.startswith("000000"):
                print("ℹ️ Dummy or Verify token detected. Skipping LINE reply.", flush=True)
                return

            try:
                reply_messages = self.process_message(user_text, user_id=user_id)
                if not isinstance(reply_messages, list):
                    reply_messages = [reply_messages]
                self.line_bot_api.reply_message(reply_token, reply_messages)
                print(f"✅ [LINE Reply Sent] Successfully replied {len(reply_messages)} message(s) to {user_id}", flush=True)
            except Exception as e:
                print(f"❌ [LINE Reply Error with Flex] {e}", flush=True)
                import traceback
                traceback.print_exc()
                try:
                    self.line_bot_api.reply_message(
                        reply_token,
                        TextSendMessage(text="ขออภัย ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง")
                    )
                    print("✅ [Fallback Error Message Sent Successfully]", flush=True)
                except Exception as e2:
                    print(f"❌ [Failed to send Fallback Reply] {e2}", flush=True)

    def process_message(
        self, user_text: str, user_id: str = "default_user"
    ) -> Union[TextSendMessage, ImageSendMessage, FlexSendMessage, List[Union[TextSendMessage, ImageSendMessage, FlexSendMessage]]]:
        """
        Process a message into plain text plus optional place cards/carousels.
        """
        classification = self.intent_classifier.classify(user_text)
        intent = classification["intent"]
        method = classification["method"]
        conf = classification["confidence"]
        
        print(f"🎯 [Intent Match] '{user_text}' -> Intent: {intent} (by {method}, conf: {conf})")

        # 1. Reset Session
        if intent == "reset":
            if user_id in self.rag_engine.sessions:
                self.rag_engine.sessions[user_id] = []
            return [TextSendMessage(text=self._welcome_text())]

        # 2. Home Navigation Hub
        if intent in ["home", "menu"]:
            return [TextSendMessage(text=self._welcome_text())]

        # 3. Explicit 2-Day Itinerary as readable text plus separate images
        if intent == "itinerary":
            return SongkhlaFlexTemplates.build_itinerary_messages(self.places)

        # 4. Explicit Food & Dessert Carousel
        if intent == "food":
            return [
                TextSendMessage(text="รวมร้านอาหาร ของหวาน และคาเฟ่ในย่านเมืองเก่าสงขลา"),
                SongkhlaFlexTemplates.build_food_carousel(self.food_places)
            ]

        # 5. Explicit Historical Attractions Carousel
        if intent == "attractions":
            return [
                TextSendMessage(text="รวมสถานที่ท่องเที่ยวและจุดเช็กอินในย่านเมืองเก่าสงขลา"),
                SongkhlaFlexTemplates.build_attractions_carousel(self.attraction_places)
            ]

        # 6. Explicit Hotels Carousel
        if intent == "hotels":
            return [
                TextSendMessage(text="รวมโรงแรมและที่พักที่มีข้อมูลในย่านเมืองเก่าสงขลา"),
                SongkhlaFlexTemplates.build_hotels_carousel(self.hotel_places)
            ]

        # 7. Emergency Contact Card
        if intent == "emergency":
            return [SongkhlaFlexTemplates.build_emergency_card()]

        # 8. Natural Language Inquiries & Deep Questions -> True Hybrid GraphRAG Engine
        print(f"🧠 [Hybrid RAG Dispatch] Query: '{user_text}'")
        rag_res = self.rag_engine.generate(query=user_text, user_id=user_id, mode="hybrid")

        answer = str(rag_res.get("answer", "")).strip()
        if not answer or answer.startswith(("Ollama Error", "Local LLM Error", "Groq Error", "Cloud API Error")):
            return [TextSendMessage(text="ขออภัย ระบบขัดข้องชั่วคราว กรุณาลองใหม่อีกครั้ง")]

        final_answer = self._append_answer_metadata(answer, rag_res)
        text_replies = [
            TextSendMessage(text=part)
            for part in self._split_line_text(final_answer)
        ]
        no_information = any(marker in answer for marker in (
            "ไม่มีข้อมูล", "ไม่พบข้อมูล", "ข้อมูลไม่เพียงพอ", "ไม่ได้ระบุ", "ไม่สามารถยืนยัน"
        ))
        is_follow_up = rag_res.get("retrieval_query", user_text.strip()) != user_text.strip()
        if no_information:
            return text_replies

        allowed_places = rag_res.get("allowed_places") or []
        if allowed_places:
            place_by_name = {
                place.get("name", "").split("(", 1)[0].strip(): place
                for place in self.places
            }
            matched_places = [
                place_by_name[name] for name in allowed_places if name in place_by_name
            ]
            if matched_places:
                return [
                    *text_replies,
                    SongkhlaFlexTemplates.build_matched_places_carousel(matched_places),
                ]

        if is_follow_up:
            return text_replies

        # Multi-Entity Resolution: Scan query and answer for places
        combined_text = f"{user_text} {answer}"
        matched_places = SongkhlaFlexTemplates.extract_matched_places(combined_text, self.places)

        # A specific place gets one place card; multi-place recommendations get a carousel.
        if matched_places:
            places_carousel = SongkhlaFlexTemplates.build_matched_places_carousel(matched_places)
            return [*text_replies, places_carousel]

        return text_replies

    @staticmethod
    def _format_source_pages(pages) -> str:
        normalized = sorted({
            page for value in (pages or [])
            if (page := SongkhlaRAGEngine._normalize_source_page(value)) is not None
        })
        if not normalized:
            return ""
        return "แหล่งข้อมูลอ้างอิง: หน้า " + ", ".join(map(str, normalized))

    @classmethod
    def _append_answer_metadata(cls, answer: str, rag_res: dict) -> str:
        """Append deterministic metadata after generation, never through the prompt."""
        footer = []
        source_line = cls._format_source_pages(rag_res.get("source_pages"))
        if source_line:
            footer.append(source_line)

        provider_display = rag_res.get("provider_display")
        if provider_display not in ("Local LLM", "Cloud LLM"):
            if rag_res.get("llm_generated_final_answer"):
                provider_display = SongkhlaRAGEngine.format_provider_label(
                    rag_res.get("provider")
                )
            else:
                provider_display = None
        if provider_display:
            footer.append(f"ประมวลผลโดย: {provider_display}")

        clean_answer = str(answer).strip()
        return clean_answer if not footer else clean_answer + "\n\n" + "\n".join(footer)

    @classmethod
    def _split_line_text(cls, text: str) -> List[str]:
        """Split safely below LINE's 5,000-character text limit."""
        raw_text = str(text)
        footer_positions = [
            position for marker in (
                "\n\nแหล่งข้อมูลอ้างอิง:", "\n\nประมวลผลโดย:"
            )
            if (position := raw_text.find(marker)) >= 0
        ]
        footer = ""
        if footer_positions:
            footer_start = min(footer_positions)
            remaining = raw_text[:footer_start].rstrip()
            footer = raw_text[footer_start:].strip()
        else:
            remaining = raw_text
        parts = []
        while len(remaining) > cls.LINE_TEXT_LIMIT:
            split_at = remaining.rfind("\n", 0, cls.LINE_TEXT_LIMIT + 1)
            if split_at <= 0:
                split_at = remaining.rfind(" ", 0, cls.LINE_TEXT_LIMIT + 1)
            if split_at <= 0:
                split_at = cls.LINE_TEXT_LIMIT
            part = remaining[:split_at].rstrip()
            if part:
                parts.append(part)
            remaining = remaining[split_at:].lstrip()
        if remaining or not parts:
            parts.append(remaining)
        if footer:
            combined = parts[-1].rstrip() + "\n\n" + footer
            if len(combined) <= cls.LINE_TEXT_LIMIT:
                parts[-1] = combined
            else:
                parts.append(footer)
        return parts

    @staticmethod
    def _welcome_text() -> str:
        return (
            "สวัสดีครับ น้องสิงขรพร้อมช่วยแนะนำย่านเมืองเก่าสงขลา\n"
            "ถามได้ทั้งสถานที่ เวลาเปิด-ปิด ราคา ร้านอาหาร ที่พัก "
            "หรือขอคำแนะนำเส้นทางท่องเที่ยวได้เลย"
        )
