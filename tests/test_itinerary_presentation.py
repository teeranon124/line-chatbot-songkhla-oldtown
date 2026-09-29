# -*- coding: utf-8 -*-
import json
import unittest
from pathlib import Path

from linebot.models import FlexSendMessage, ImageSendMessage, TextSendMessage

from src.flex_templates import PLACE_IMAGES, SongkhlaFlexTemplates
from src.line_handler import SongkhlaLineHandler


ROOT = Path(__file__).resolve().parents[1]
PLACES = json.loads((ROOT / "data" / "songkhla_places_facts.json").read_text(encoding="utf-8"))


class _Classifier:
    def __init__(self, intent):
        self.intent = intent

    def classify(self, _text):
        return {"intent": self.intent, "method": "test", "confidence": 1.0}


class _RAG:
    sessions = {}

    def __init__(self, answer):
        self.answer = answer

    def generate(self, query, user_id, mode):
        return {"answer": self.answer, "retrieval_query": query}


def _handler(intent, rag=None):
    handler = SongkhlaLineHandler.__new__(SongkhlaLineHandler)
    handler.intent_classifier = _Classifier(intent)
    handler.rag_engine = rag or _RAG("คำตอบปกติ")
    handler.places = PLACES
    return handler


class ItineraryPresentationTests(unittest.TestCase):
    def test_one_day_returns_text_then_optional_image(self):
        messages = SongkhlaFlexTemplates.build_itinerary_messages(PLACES, [1])
        self.assertEqual([type(message) for message in messages], [TextSendMessage, ImageSendMessage])

    def test_two_days_preserve_day_order_within_reply_limit(self):
        messages = SongkhlaFlexTemplates.build_itinerary_messages(PLACES)
        self.assertEqual(len(messages), 4)
        self.assertIn("วันที่ 1", messages[0].text)
        self.assertIn("วันที่ 2", messages[2].text)

    def test_itinerary_contains_no_flex_message(self):
        messages = SongkhlaFlexTemplates.build_itinerary_messages(PLACES)
        self.assertFalse(any(isinstance(message, FlexSendMessage) for message in messages))

    def test_original_itinerary_wording_is_preserved(self):
        text = SongkhlaFlexTemplates.build_itinerary_messages(PLACES, [1])[0].text
        self.assertIn("09.00 น. หอศิลป์สงขลา (Songkhla Art Center)", text)
        self.assertIn("18.00 น. มื้อค่ำ: ต้มยำแห้ง ร้านแต้เฮี้ยงอิ้ว", text)

    def test_map_links_are_appended_after_day_content(self):
        text = SongkhlaFlexTemplates.build_itinerary_messages(PLACES, [1])[0].text
        self.assertGreater(text.index("📍 แผนที่"), text.index("18.00 น."))

    def test_map_urls_are_exact_structured_facts_and_deduplicated(self):
        duplicate = dict(next(place for place in PLACES if place["id"] == "aitim_oang"))
        duplicate["google_maps_url"] = next(
            place["google_maps_url"] for place in PLACES if place["id"] == "songkhla_street_art"
        )
        places = [place for place in PLACES if place["id"] != "aitim_oang"] + [duplicate]
        text = SongkhlaFlexTemplates.build_itinerary_messages(places, [1])[0].text
        self.assertEqual(text.count(duplicate["google_maps_url"]), 1)
        exact_url = next(place["google_maps_url"] for place in PLACES if place["id"] == "kiat_fang")
        self.assertIn(exact_url, text)
        unrelated_url = next(place["google_maps_url"] for place in PLACES if place["id"] == "hotel_club_tree")
        self.assertNotIn(unrelated_url, text)

    def test_missing_map_is_omitted_without_fallback_url(self):
        places = [dict(place) for place in PLACES]
        target = next(place for place in places if place["id"] == "khao_tang_kuan")
        target.pop("google_maps_url")
        text = SongkhlaFlexTemplates.build_itinerary_messages(places, [2])[0].text
        self.assertIn("เขาตังกวน", text)
        self.assertNotIn("Songkhla+Old+Town", text)
        self.assertNotIn("• เขาตังกวน\nhttp", text)

    def test_image_uses_existing_project_url_as_separate_message(self):
        messages = SongkhlaFlexTemplates.build_itinerary_messages(PLACES, [2])
        self.assertEqual(messages[1].original_content_url, PLACE_IMAGES["singora_tram"])
        self.assertEqual(messages[1].preview_image_url, PLACE_IMAGES["singora_tram"])

    def test_missing_image_keeps_itinerary_without_fake_image(self):
        image_url = PLACE_IMAGES.pop("songkhla_art_center")
        try:
            messages = SongkhlaFlexTemplates.build_itinerary_messages(PLACES, [1])
        finally:
            PLACE_IMAGES["songkhla_art_center"] = image_url
        self.assertEqual(len(messages), 1)
        self.assertIsInstance(messages[0], TextSendMessage)

    def test_handler_itinerary_uses_new_presentation(self):
        messages = _handler("itinerary").process_message("ขอแผนเที่ยว 2 วัน 1 คืน")
        self.assertEqual(len(messages), 4)
        self.assertFalse(any(isinstance(message, FlexSendMessage) for message in messages))

    def test_normal_place_and_insufficient_information_responses_remain_unchanged(self):
        normal = _handler("qa", _RAG("ร้านไอติมโอ่งเปิดเวลา 10:00 - 18:30 น.")).process_message(
            "ร้านไอติมโอ่งเปิดกี่โมง"
        )
        insufficient = _handler("qa", _RAG("ไม่มีข้อมูลเพียงพอในฐานความรู้")).process_message("ถามข้อมูลที่ไม่มี")
        self.assertEqual(len(normal), 2)
        self.assertIsInstance(normal[0], TextSendMessage)
        self.assertIsInstance(normal[1], FlexSendMessage)
        self.assertEqual(len(insufficient), 1)
        self.assertEqual(insufficient[0].text, "ไม่มีข้อมูลเพียงพอในฐานความรู้")


if __name__ == "__main__":
    unittest.main()
