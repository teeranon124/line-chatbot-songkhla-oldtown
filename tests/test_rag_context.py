# -*- coding: utf-8 -*-
import sys
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Keep these regression tests independent of FAISS, SentenceTransformers,
# PyThaiNLP, NetworkX, dotenv, and every external service.
if "faiss" not in sys.modules:
    sys.modules["faiss"] = types.ModuleType("faiss")

try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules["numpy"] = types.ModuleType("numpy")

pythainlp_module = types.ModuleType("pythainlp")
pythainlp_tokenize_module = types.ModuleType("pythainlp.tokenize")
pythainlp_tokenize_module.word_tokenize = lambda text, engine=None: text.split()
sys.modules.setdefault("pythainlp", pythainlp_module)
sys.modules.setdefault("pythainlp.tokenize", pythainlp_tokenize_module)

sentence_transformers_module = types.ModuleType("sentence_transformers")
sentence_transformers_module.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", sentence_transformers_module)

dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_module)

requests_module = types.ModuleType("requests")
requests_module.post = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("External HTTP must not be used by these tests")
)
sys.modules.setdefault("requests", requests_module)

graph_engine_module = types.ModuleType("src.graph_engine")
graph_engine_module.SongkhlaGraphEngine = object
sys.modules.setdefault("src.graph_engine", graph_engine_module)

from src.rag_engine import SongkhlaRAGEngine
from src.retriever import HybridRetriever
from src.line_handler import SongkhlaLineHandler


TEXT_CHUNK = {
    "chunk_id": "chunk_014",
    "title": "ร้านไอติมโอ่ง (AnyFlip หน้า 24)",
    "source_page": 24,
    "place_id": "aitim_oang",
    "category": "ของหวาน / เครื่องดื่ม",
    "content": "ร้านไอติมโอ่งตั้งอยู่บนถนนนางงาม เปิดเวลา 10:00 - 18:30 น."
}

GRAPH_CONTENT = (
    "โครงสร้างความสัมพันธ์ใน Knowledge Graph สำหรับ: ร้านไอติมโอ่ง\n"
    "- โครงข่ายความสัมพันธ์เชื่อมโยงออกไป: LOCATED_ON -> ถนนนางงาม"
)

GRAPH_CHUNK = {
    "chunk_id": "graph_aitim_oang",
    "title": "[Graph Subgraph] ร้านไอติมโอ่ง",
    "category": "Knowledge Graph",
    "content": GRAPH_CONTENT,
    "score": 2.5
}


class FakeRankedRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.last_query = None

    def retrieve(self, query, top_k=None, mode="hybrid"):
        self.last_query = query
        return self.chunks


class FakeSearchBackend:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=4):
        return self.results[:top_k]


class FakeGraphBackend:
    def __init__(self, results):
        self.results = results

    def search_subgraph(self, query, top_k=3):
        return self.results[:top_k]


def make_engine(chunks):
    engine = SongkhlaRAGEngine.__new__(SongkhlaRAGEngine)
    engine.retriever = FakeRankedRetriever(chunks)
    engine.groq_api_key = ""
    engine.groq_model = "fake-cloud"
    engine.ollama_base_url = "http://not-used"
    engine.local_model = "fake-local"
    engine.sessions = {}
    return engine


class ContextContractTests(unittest.TestCase):
    def test_text_only_context(self):
        context = SongkhlaRAGEngine.build_context([dict(TEXT_CHUNK)])

        self.assertIn("[หลักฐานข้อความ]", context)
        self.assertIn(TEXT_CHUNK["content"], context)
        self.assertIn("หน้า: 24", context)
        self.assertNotIn("[หลักฐานกราฟ]", context)

    def test_graph_only_context(self):
        context = SongkhlaRAGEngine.build_context([dict(GRAPH_CHUNK)])

        self.assertIn("[หลักฐานกราฟ]", context)
        self.assertIn("LOCATED_ON -> ถนนนางงาม", context)
        self.assertIn("Graph Chunk ID: graph_aitim_oang", context)
        self.assertNotIn("[หลักฐานข้อความ]", context)

    def test_hybrid_text_and_graph_context(self):
        hybrid_chunk = dict(TEXT_CHUNK)
        hybrid_chunk["graph_evidence"] = [dict(GRAPH_CHUNK)]

        context = SongkhlaRAGEngine.build_context([hybrid_chunk])

        self.assertIn("[หลักฐานข้อความ]", context)
        self.assertIn(TEXT_CHUNK["content"], context)
        self.assertIn("[หลักฐานกราฟ]", context)
        self.assertIn(GRAPH_CONTENT, context)

    def test_no_graph_hallucination(self):
        context = SongkhlaRAGEngine.build_context([dict(TEXT_CHUNK)])

        self.assertNotIn("LOCATED_ON", context)
        self.assertNotIn("ความสัมพันธ์ที่ค้นคืน", context)

    def test_duplicate_graph_content_is_rendered_once(self):
        duplicate = dict(TEXT_CHUNK)
        duplicate["content"] = GRAPH_CONTENT
        duplicate["graph_evidence"] = [dict(GRAPH_CHUNK)]
        duplicate["graph_subgraph"] = GRAPH_CONTENT

        context = SongkhlaRAGEngine.build_context([duplicate])

        self.assertEqual(context.count(GRAPH_CONTENT), 1)
        self.assertEqual(context.count("[หลักฐานข้อความ]"), 1)
        self.assertNotIn("[หลักฐานกราฟ]", context)

    def test_legacy_hybrid_result_remains_compatible(self):
        legacy = dict(TEXT_CHUNK)
        legacy["graph_subgraph"] = GRAPH_CONTENT

        context = SongkhlaRAGEngine.build_context([legacy])

        self.assertIn(TEXT_CHUNK["content"], context)
        self.assertIn(GRAPH_CONTENT, context)

    def test_retrieval_modes_keep_compatible_result_shapes(self):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.chunks = [dict(TEXT_CHUNK)]
        retriever.dense = FakeSearchBackend([{"chunk_index": 0, "score": 0.9}])
        retriever.sparse = FakeSearchBackend([{"chunk_index": 0, "score": 4.2}])
        retriever.graph = FakeGraphBackend([dict(GRAPH_CHUNK)])

        dense = retriever.retrieve("ไอติมโอ่ง", top_k=1, mode="dense")
        sparse = retriever.retrieve("ไอติมโอ่ง", top_k=1, mode="sparse")
        graph = retriever.retrieve("ไอติมโอ่ง", top_k=1, mode="graph")
        hybrid = retriever.retrieve("ไอติมโอ่ง", top_k=1, mode="hybrid")

        self.assertEqual(dense[0]["chunk_id"], "chunk_014")
        self.assertEqual(sparse[0]["chunk_id"], "chunk_014")
        self.assertEqual(graph[0]["chunk_id"], "graph_aitim_oang")
        self.assertEqual(hybrid[0]["content"], TEXT_CHUNK["content"])
        self.assertEqual(hybrid[0]["graph_evidence"][0]["content"], GRAPH_CONTENT)
        self.assertEqual(hybrid[0]["graph_subgraph"], GRAPH_CONTENT)
        self.assertNotIn("graph_evidence", retriever.chunks[0])

    def test_graph_evidence_reaches_actual_llm_prompt_boundary(self):
        hybrid_chunk = dict(TEXT_CHUNK)
        hybrid_chunk["graph_evidence"] = [dict(GRAPH_CHUNK)]
        engine = make_engine([hybrid_chunk])
        captured = {}

        def fake_ollama(messages, model_name):
            captured["messages"] = messages
            captured["model_name"] = model_name
            return "คำตอบจำลอง"

        engine.call_ollama = fake_ollama
        result = engine.generate(
            query="ร้านไอติมโอ่งอยู่ถนนอะไร",
            mode="hybrid",
            target_llm="ollama",
            user_id=""
        )

        final_user_prompt = captured["messages"][-1]["content"]
        self.assertIn(TEXT_CHUNK["content"], final_user_prompt)
        self.assertIn("LOCATED_ON -> ถนนนางงาม", final_user_prompt)
        self.assertEqual(captured["model_name"], "fake-local")
        self.assertEqual(result["answer"], "คำตอบจำลอง")


class FollowUpAndGroundingTests(unittest.TestCase):
    def _generate_with_history(self, place_name, follow_up):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine._known_place_names = ["ร้านไอติมโอ่ง", "เขาตังกวน"]
        engine.sessions["traveler"] = [
            {"role": "user", "content": f"ขอข้อมูล{place_name}"},
            {"role": "assistant", "content": f"ข้อมูลของ{place_name}"},
        ]
        engine.call_ollama = lambda messages, model_name: "คำตอบจำลอง"
        result = engine.generate(
            query=follow_up,
            mode="hybrid",
            target_llm="ollama",
            user_id="traveler"
        )
        return engine, result

    def test_follow_up_retains_aitim_oang_referent(self):
        engine, result = self._generate_with_history(
            "ร้านไอติมโอ่ง", "แล้วเปิดกี่โมง"
        )

        self.assertEqual(engine.retriever.last_query, "ร้านไอติมโอ่ง เปิดกี่โมง")
        self.assertEqual(result["retrieval_query"], "ร้านไอติมโอ่ง เปิดกี่โมง")

    def test_follow_up_retains_khao_tang_kuan_referent(self):
        engine, result = self._generate_with_history(
            "เขาตังกวน", "แล้วค่าเข้าเท่าไหร่"
        )

        self.assertEqual(engine.retriever.last_query, "เขาตังกวน ค่าเข้าเท่าไหร่")
        self.assertEqual(result["retrieval_query"], "เขาตังกวน ค่าเข้าเท่าไหร่")

    def test_exact_numeric_evidence_reaches_prompt_unchanged(self):
        numeric_chunk = dict(TEXT_CHUNK)
        numeric_chunk["content"] = (
            "เปิด 10:00 - 18:30 น. ราคา 20 - 30 บาท "
            "และอยู่ห่าง 250 เมตร"
        )
        engine = make_engine([numeric_chunk])
        captured = {}

        def fake_ollama(messages, model_name):
            captured["prompt"] = messages[-1]["content"]
            return "เปิด 10:00 - 18:30 น. ราคา 20 - 30 บาท ห่าง 250 เมตร"

        engine.call_ollama = fake_ollama
        engine.generate("ขอเวลา ราคา และระยะทาง", target_llm="ollama", user_id="")

        for exact_value in ("10:00 - 18:30", "20 - 30 บาท", "250 เมตร"):
            self.assertIn(exact_value, captured["prompt"])

    def test_application_does_not_add_unsupported_place_to_prompt(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        captured = {}

        def fake_ollama(messages, model_name):
            captured["prompt"] = messages[-1]["content"]
            return "ร้านไอติมโอ่งเปิด 10:00 - 18:30 น."

        engine.call_ollama = fake_ollama
        engine.generate("ร้านไอติมโอ่งเปิดกี่โมง", target_llm="ollama", user_id="")

        self.assertNotIn("ร้านเจ๊นิ", captured["prompt"])
        self.assertNotIn("เขาตังกวน", captured["prompt"])


class MultiEntityFollowUpTests(unittest.TestCase):
    @staticmethod
    def _history(user_text, assistant_text="คำตอบก่อนหน้า"):
        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    def test_single_entity_follow_up_remains_supported(self):
        engine = make_engine([])
        resolved = engine.resolve_retrieval_query(
            "แล้วราคาเท่าไหร่",
            self._history("ร้านไอติมโอ่งเปิดกี่โมง", "ร้านไอติมโอ่งเปิดทุกวัน"),
        )
        self.assertEqual(resolved, "ร้านไอติมโอ่ง ราคาเท่าไหร่")

    def test_two_entity_route_follow_up_keeps_both_places(self):
        engine = make_engine([])
        resolved = engine.resolve_retrieval_query(
            "ควรไปไหนก่อน",
            self._history("ร้านไอติมโอ่งกับบ้านจีน 300 ปีไกลกันไหม"),
        )
        self.assertEqual(
            resolved, "ร้านไอติมโอ่ง บ้านจีน 300 ปี ควรไปไหนก่อน"
        )

    def test_three_entity_route_follow_up_keeps_all_places(self):
        engine = make_engine([])
        resolved = engine.resolve_retrieval_query(
            "ควรไปที่ไหนก่อน-หลัง",
            self._history(
                "อยากเที่ยวโรงสีแดง บ้านจีน 300 ปี และร้านไอติมโอ่ง"
            ),
        )
        self.assertEqual(
            resolved,
            "โรงสีแดง หับโห้หิ้น บ้านจีน 300 ปี ร้านไอติมโอ่ง "
            "ควรไปที่ไหนก่อน-หลัง",
        )

    def test_assistant_introduced_entity_resolves_shop_reference(self):
        engine = make_engine([])
        resolved = engine.resolve_retrieval_query(
            "แล้วร้านนี้เปิดกี่โมง",
            self._history(
                "แนะนำของหวานงบ 30 บาท",
                "แนะนำร้านไอติมโอ่ง ราคา 20 - 30 บาทครับ",
            ),
        )
        self.assertEqual(resolved, "ร้านไอติมโอ่ง เปิดกี่โมง")

    def test_explicit_new_place_does_not_inherit_old_entities(self):
        engine = make_engine([])
        query = "เขาตังกวนเปิดกี่โมง"
        resolved = engine.resolve_retrieval_query(
            query,
            self._history("ร้านไอติมโอ่งเปิดกี่โมง"),
        )
        self.assertEqual(resolved, query)
        self.assertNotIn("ร้านไอติมโอ่ง", resolved)

    def test_aliases_resolve_to_one_canonical_place(self):
        engine = make_engine([])
        resolved = engine.resolve_retrieval_query(
            "แล้วไปไหนต่อ",
            self._history("โรงสีแดง และโรงสีแดง หับโห้หิ้น"),
        )
        self.assertEqual(resolved.count("โรงสีแดง หับโห้หิ้น"), 1)
        self.assertEqual(resolved, "โรงสีแดง หับโห้หิ้น ไปไหนต่อ")

    def test_route_prompt_guards_self_relations_and_unsupported_order(self):
        self_relation_chunk = dict(GRAPH_CHUNK)
        self_relation_chunk["content"] = (
            "ร้านไอติมโอ่ง -> NEARBY -> ร้านไอติมโอ่ง"
        )
        engine = make_engine([self_relation_chunk])
        engine.sessions["traveler"] = self._history(
            "ร้านไอติมโอ่งกับบ้านจีน 300 ปีไกลกันไหม"
        )
        captured = {}

        def fake_ollama(messages, model_name):
            captured["prompt"] = messages[-1]["content"]
            return "ควรไปร้านไอติมโอ่ง ซึ่งอยู่ใกล้กับร้านไอติมโอ่ง"

        engine.call_ollama = fake_ollama
        result = engine.generate(
            "ควรไปไหนก่อน", target_llm="ollama", user_id="traveler"
        )

        self.assertIn("ห้ามกล่าวว่าสถานที่อยู่ใกล้ตัวเอง", captured["prompt"])
        self.assertIn("ห้ามแต่งระยะทางหรือเหตุผล", captured["prompt"])
        self.assertIn("ร้านไอติมโอ่ง บ้านจีน 300 ปี", captured["prompt"])
        self.assertNotIn("อยู่ใกล้กับร้านไอติมโอ่ง", result["answer"])
        self.assertEqual(result["allowed_places"], ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"])


class RoutePlanningTests(unittest.TestCase):
    def test_route_answer_drops_llm_appended_unsupported_note(self):
        engine = make_engine([])
        engine.call_ollama = lambda messages, model_name: (
            "1. โรงสีแดง หับโห้หิ้น\n"
            "2. บ้านจีน 300 ปี\n"
            "3. ร้านไอติมโอ่ง\n"
            "ระยะห่างโดยประมาณแบบเส้นตรง 219 เมตร และ 390 เมตร\n"
            "หมายเหตุ: ข้อมูลเวลาเปิด-ปิดไม่ปรากฏในหลักฐาน"
        )
        result = engine.generate(
            "อยากเที่ยวโรงสีแดง บ้านจีน 300 ปี และร้านไอติมโอ่ง "
            "ควรไปที่ไหนก่อน-หลัง",
            target_llm="ollama",
            user_id="",
        )

        self.assertEqual(result["answer"], result["route_plan"]["answer"])
        self.assertNotIn("หมายเหตุ", result["answer"])
        self.assertNotIn("ไม่ปรากฏในหลักฐาน", result["answer"])

    def test_direct_distance_question_rejects_misattributed_250_metres(self):
        engine = make_engine([])
        engine.call_ollama = lambda messages, model_name: (
            "ร้านไอติมโอ่งอยู่ห่างจากบ้านจีน 300 ปี ประมาณ 250 เมตร"
        )
        result = engine.generate(
            "ร้านไอติมโอ่งอยู่ห่างจากบ้านจีน 300 ปี exactly กี่เมตร",
            target_llm="ollama",
            user_id="",
        )
        self.assertIn("ระยะห่างโดยประมาณแบบเส้นตรง 390 เมตร", result["answer"])
        self.assertNotIn("250 เมตร", result["answer"])
        self.assertEqual(
            result["allowed_places"], ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"]
        )

    def test_two_place_follow_up_is_locked_and_includes_distance(self):
        engine = make_engine([])
        engine.sessions["traveler"] = [
            {"role": "user", "content": "ร้านไอติมโอ่งกับบ้านจีน 300 ปีไกลกันไหม"},
            {"role": "assistant", "content": "ทั้งสองแห่งอยู่ในย่านเมืองเก่าสงขลา"},
        ]
        engine.call_ollama = lambda messages, model_name: "1. บ้านจีน 300 ปี\n2. ร้านไอติมโอ่ง"

        result = engine.generate(
            "ควรไปไหนก่อน", target_llm="ollama", user_id="traveler"
        )

        self.assertEqual(result["allowed_places"], ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"])
        self.assertEqual(
            result["route_plan"]["ordered_places"],
            ["บ้านจีน 300 ปี", "ร้านไอติมโอ่ง"],
        )
        self.assertIn("ระยะห่างโดยประมาณแบบเส้นตรง 390 เมตร", result["answer"])
        self.assertNotIn("โรงสีแดง", result["answer"])

    def test_three_place_route_is_deterministic_and_constrained(self):
        engine = make_engine([])
        engine.call_ollama = lambda messages, model_name: "แนะนำโรงแรมคลับทรีก่อน"
        result = engine.generate(
            "อยากเที่ยวโรงสีแดง บ้านจีน 300 ปี และร้านไอติมโอ่ง "
            "ควรไปที่ไหนก่อน-หลัง",
            target_llm="ollama",
            user_id="",
        )

        expected = ["โรงสีแดง หับโห้หิ้น", "บ้านจีน 300 ปี", "ร้านไอติมโอ่ง"]
        self.assertEqual(result["allowed_places"], expected)
        self.assertEqual(result["route_plan"]["ordered_places"], expected)
        self.assertNotIn("โรงแรมคลับทรี", result["answer"])
        self.assertIn("219 เมตร", result["answer"])
        self.assertIn("390 เมตร", result["answer"])

    def test_same_place_set_in_different_input_orders_has_same_route(self):
        engine = make_engine([])
        queries = [
            "อยากเที่ยวโรงสีแดง บ้านจีน 300 ปี และร้านไอติมโอ่ง ควรไปที่ไหนก่อน-หลัง",
            "ร้านไอติมโอ่ง โรงสีแดง บ้านจีน 300 ปี ไปที่ไหนก่อนดี",
            "บ้านจีน 300 ปี ร้านไอติมโอ่ง โรงสีแดง ช่วยจัดลำดับให้หน่อย",
        ]
        routes = []
        for query in queries:
            places = engine._extract_place_entities(query)
            routes.append(engine._build_route_plan(places, [], query)["ordered_places"])
        self.assertEqual(routes[0], routes[1])
        self.assertEqual(routes[1], routes[2])
        self.assertEqual(
            routes[0],
            ["โรงสีแดง หับโห้หิ้น", "บ้านจีน 300 ปี", "ร้านไอติมโอ่ง"],
        )

    def test_explicit_start_constraint_remains_first(self):
        engine = make_engine([])
        query = (
            "เริ่มจากร้านไอติมโอ่ง แล้วเที่ยวโรงสีแดงกับบ้านจีน 300 ปี "
            "ช่วยจัดลำดับให้หน่อย"
        )
        places = engine._extract_place_entities(query)
        plan = engine._build_route_plan(places, [], query)
        self.assertEqual(plan["ordered_places"][0], "ร้านไอติมโอ่ง")
        self.assertIn("ผู้ใช้ระบุให้เริ่มจากร้านไอติมโอ่ง", plan["decisions"])

    def test_explicit_end_constraint_remains_last(self):
        engine = make_engine([])
        query = (
            "เที่ยวร้านไอติมโอ่ง บ้านจีน 300 ปี และโรงสีแดง "
            "โดยปิดท้ายที่โรงสีแดง ช่วยจัดลำดับให้หน่อย"
        )
        places = engine._extract_place_entities(query)
        plan = engine._build_route_plan(places, [], query)
        self.assertEqual(plan["ordered_places"][-1], "โรงสีแดง หับโห้หิ้น")
        self.assertIn("ผู้ใช้ระบุให้ปิดท้ายที่โรงสีแดง หับโห้หิ้น", plan["decisions"])

    def test_route_reason_uses_only_structured_facts(self):
        engine = make_engine([])
        places = ["ร้านไอติมโอ่ง", "โรงสีแดง หับโห้หิ้น", "บ้านจีน 300 ปี"]
        plan = engine._build_route_plan(places, [], "ช่วยจัดลำดับให้หน่อย")
        explanation = " ".join(plan["decisions"])
        self.assertIn("สถานที่ท่องเที่ยวทางประวัติศาสตร์", explanation)
        self.assertIn("08:00 - 18:00 น.", explanation)
        self.assertIn("ของหวาน / เครื่องดื่ม", explanation)
        self.assertNotIn("สถานที่แรกที่ผู้ใช้ระบุ", plan["answer"])
        self.assertNotIn("nearest-neighbor", plan["answer"])

    def test_explicit_distance_wins_over_haversine(self):
        engine = make_engine([])
        explicit = {
            "content": "ร้านไอติมโอ่ง ↔ บ้านจีน 300 ปี ระยะทาง 250 เมตร"
        }
        distance = engine.resolve_pairwise_distance(
            "ร้านไอติมโอ่ง", "บ้านจีน 300 ปี", [explicit]
        )
        self.assertEqual(distance["source"], "explicit")
        self.assertEqual(distance["meters"], 250)
        self.assertEqual(distance["description"], "ระยะทางตามหลักฐาน 250 เมตร")

    def test_haversine_uses_lat_lon_and_is_labelled_straight_line(self):
        engine = make_engine([])
        distance = engine.resolve_pairwise_distance(
            "ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"
        )
        self.assertEqual(distance["source"], "haversine")
        self.assertEqual(distance["meters"], 390)
        self.assertEqual(
            distance["description"], "ระยะห่างโดยประมาณแบบเส้นตรง 390 เมตร"
        )
        self.assertNotIn("เดิน", distance["description"])
        self.assertNotIn("ขับ", distance["description"])

    def test_missing_evidence_and_coordinates_returns_unknown(self):
        engine = make_engine([])
        engine._place_records = {"ก": {"name": "ก"}, "ข": {"name": "ข"}}
        distance = engine.resolve_pairwise_distance("ก", "ข")
        self.assertEqual(distance["source"], "unknown")
        self.assertIsNone(distance["meters"])
        self.assertNotRegex(distance["description"], r"\d+\s*เมตร")

    def test_line_cards_use_only_route_allowed_places(self):
        class Classifier:
            @staticmethod
            def classify(_text):
                return {"intent": "qa_hybrid", "method": "test", "confidence": 1.0}

        class RAG:
            sessions = {}

            @staticmethod
            def generate(query, user_id, mode):
                return {
                    "answer": "1. ร้านไอติมโอ่ง\n2. บ้านจีน 300 ปี",
                    "retrieval_query": "ร้านไอติมโอ่ง บ้านจีน 300 ปี ควรไปไหนก่อน",
                    "allowed_places": ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"],
                }

        import json
        from src.config import paths

        handler = SongkhlaLineHandler.__new__(SongkhlaLineHandler)
        handler.intent_classifier = Classifier()
        handler.rag_engine = RAG()
        with open(paths.facts_path, "r", encoding="utf-8") as f:
            handler.places = json.load(f)

        messages = handler.process_message("ควรไปไหนก่อน", user_id="traveler")
        carousel = messages[1].contents
        card_names = [
            bubble.body.contents[0].text for bubble in carousel.contents
        ]
        self.assertEqual(card_names, ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"])

    def test_single_place_follow_up_still_resolves_without_route_plan(self):
        engine = make_engine([])
        history = [
            {"role": "user", "content": "ร้านไอติมโอ่งเปิดกี่โมง"},
            {"role": "assistant", "content": "เปิด 10:00 - 18:30 น."},
        ]
        self.assertEqual(
            engine.resolve_retrieval_query("แล้วราคาเท่าไหร่", history),
            "ร้านไอติมโอ่ง ราคาเท่าไหร่",
        )


class EvidenceRelevanceGuardrailTests(unittest.TestCase):
    def test_known_khao_tang_kuan_lift_price_uses_existing_fact(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: (
            "ข้อมูลที่เกี่ยวข้องไม่ได้ระบุถึงราคาของลิฟต์ขึ้นเขาตังกวน"
        )
        result = engine.generate(
            "ค่าลิฟต์ขึ้นเขาตังกวนราคาเท่าไร",
            target_llm="ollama",
            user_id="",
        )

        self.assertEqual(result["guardrail_status"], "PASS")
        self.assertIn("ผู้ใหญ่ 30 บาท / เด็ก 20 บาท", result["answer"])
        self.assertNotIn("ไม่ได้ระบุ", result["answer"])

    def test_natural_thai_out_of_scope_destination_abstains(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("อยากไปเชียงใหม่ต้องไปยังไง", user_id="")

        self.assertEqual(result["guardrail_status"], "ABSTAIN")
        self.assertEqual(
            result["answer"],
            "ขออภัยครับ ตอนนี้ระบบยังไม่มีข้อมูลเกี่ยวกับเชียงใหม่ "
            "จึงยังไม่สามารถแนะนำเส้นทางไปเชียงใหม่ได้ครับ",
        )
        for unrelated in ("สงขลา", "โปรแกรมท่องเที่ยว", "จังหวัดอื่น"):
            self.assertNotIn(unrelated, result["answer"])

    def test_llm_abstention_cannot_continue_with_unrelated_recommendations(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: (
            "ขออภัยครับ จากข้อมูลที่มีตอนนี้ยังไม่พบข้อมูลเกี่ยวกับโรงสีแดง "
            "แต่แนะนำโรงแรมมอนทาน่า รถราง และโทร 074-311015"
        )
        result = engine.generate(
            "ขอเส้นทางท่องเที่ยวที่ข้อมูลยังไม่รองรับ",
            target_llm="ollama",
            user_id="",
        )

        self.assertEqual(
            result["answer"],
            "ขออภัยครับ จากข้อมูลที่มีตอนนี้ยังไม่พบข้อมูลที่เกี่ยวข้องครับ",
        )
        self.assertNotIn("โรงแรมมอนทาน่า", result["answer"])
        self.assertNotIn("รถราง", result["answer"])
        self.assertNotIn("074-311015", result["answer"])

    def test_unknown_entity_route_abstains_without_llm(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("ไปบ้านพรุไปยังไง", user_id="")

        self.assertEqual(result["guardrail_status"], "ABSTAIN")
        self.assertIn("บ้านพรุ", result["answer"])
        for unrelated in ("ร้านไอติมโอ่ง", "โรงสีแดง", "บ้านนครใน"):
            self.assertNotIn(unrelated, result["answer"])

    def test_unknown_english_entity_opening_hours_abstains(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("Cream World เปิดกี่โมง", user_id="")

        self.assertEqual(result["guardrail_status"], "ABSTAIN")
        self.assertIn("Cream World", result["answer"])
        self.assertIn("เวลาเปิด-ปิด", result["answer"])

    def test_known_entity_continues_to_generation(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        called = {"value": False}

        def fake_ollama(messages, model_name):
            called["value"] = True
            return "ร้านไอติมโอ่งเปิดเวลา 10:00 - 18:30 น."

        engine.call_ollama = fake_ollama
        result = engine.generate(
            "ร้านไอติมโอ่งเปิดกี่โมง", target_llm="ollama", user_id=""
        )
        self.assertTrue(called["value"])
        self.assertEqual(result["guardrail_status"], "PASS")

    def test_generic_discovery_is_not_blocked(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: "คำแนะนำจากหลักฐาน"
        result = engine.generate(
            "มีที่เที่ยวอะไรแนะนำบ้าง", target_llm="ollama", user_id=""
        )
        self.assertEqual(result["guardrail_status"], "PASS")

    def test_generic_constrained_recommendation_is_not_blocked(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: "ร้านไอติมโอ่ง ราคา 20 - 30 บาท"
        result = engine.generate(
            "แนะนำของหวานงบไม่เกิน 30 บาท", target_llm="ollama", user_id=""
        )
        self.assertEqual(result["guardrail_status"], "PASS")

    def test_known_entity_missing_parking_returns_partial_without_llm(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("ร้านไอติมโอ่งมีที่จอดรถไหม", user_id="")

        self.assertEqual(result["guardrail_status"], "PARTIAL")
        self.assertIn("ที่จอดรถ", result["answer"])
        self.assertIn("ร้านไอติมโอ่ง", result["answer"])

    def test_mixed_multi_entity_abstains_for_unknown_without_distance(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("ร้านไอติมโอ่งกับบ้านพรุไกลกันไหม", user_id="")

        self.assertEqual(result["guardrail_status"], "ABSTAIN")
        self.assertIn("บ้านพรุ", result["answer"])
        self.assertIn("ร้านไอติมโอ่ง", result["answer"])
        self.assertNotRegex(result["answer"], r"\d+\s*เมตร")

    def test_unknown_entity_line_response_has_text_only(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")

        class Classifier:
            @staticmethod
            def classify(_text):
                return {"intent": "qa_hybrid", "method": "test", "confidence": 1.0}

        import json
        from src.config import paths

        handler = SongkhlaLineHandler.__new__(SongkhlaLineHandler)
        handler.intent_classifier = Classifier()
        handler.rag_engine = engine
        with open(paths.facts_path, "r", encoding="utf-8") as f:
            handler.places = json.load(f)

        messages = handler.process_message("ไปบ้านพรุไปยังไง", user_id="traveler")
        self.assertEqual(len(messages), 1)
        self.assertIn("บ้านพรุ", messages[0].text)


class AnswerMetadataTests(unittest.TestCase):
    @staticmethod
    def _line_text(result):
        return SongkhlaLineHandler._append_answer_metadata(result["answer"], result)

    def test_local_factual_response_has_real_page_and_local_provider(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: (
            "ร้านไอติมโอ่งเปิดเวลา 10:00 - 18:30 น.ครับ"
        )
        result = engine.generate(
            "ร้านไอติมโอ่งเปิดกี่โมง", target_llm="ollama", user_id=""
        )
        text = self._line_text(result)
        self.assertEqual(result["source_pages"], [24])
        self.assertIn("แหล่งข้อมูลอ้างอิง: หน้า 24", text)
        self.assertTrue(text.endswith("ประมวลผลโดย: Local LLM"))

    def test_cloud_provider_label(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.groq_api_key = "configured-for-test"
        engine.call_groq = lambda messages, model_name: "คำตอบจาก cloud เกี่ยวกับร้านไอติมโอ่ง"
        result = engine.generate(
            "ร้านไอติมโอ่งเปิดกี่โมง", target_llm="cloud", user_id=""
        )
        self.assertIn("ประมวลผลโดย: Cloud LLM", self._line_text(result))

    def test_cloud_failover_displays_local_provider(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.groq_api_key = "configured-for-test"
        engine.call_groq = lambda messages, model_name: "Cloud API Error: timeout"
        engine.call_ollama = lambda messages, model_name: "คำตอบ local ร้านไอติมโอ่ง"
        result = engine.generate(
            "ร้านไอติมโอ่งเปิดกี่โมง", target_llm="cloud", user_id=""
        )
        self.assertEqual(result["provider"], "ollama (failover)")
        self.assertIn("ประมวลผลโดย: Local LLM", self._line_text(result))

    def test_duplicate_pages_are_deduplicated(self):
        pages = SongkhlaRAGEngine.extract_source_pages(
            [dict(TEXT_CHUNK), dict(TEXT_CHUNK), dict(TEXT_CHUNK)],
            ["ร้านไอติมโอ่ง"],
        )
        footer = SongkhlaLineHandler._format_source_pages(pages)
        self.assertEqual(footer, "แหล่งข้อมูลอ้างอิง: หน้า 24")
        self.assertEqual(footer.count("24"), 1)

    def test_multiple_pages_are_sorted(self):
        second = {
            "chunk_id": "chunk_009",
            "title": "บ้านจีน 300 ปี",
            "source_page": 26,
            "content": "ข้อมูลบ้านจีน 300 ปี",
        }
        pages = SongkhlaRAGEngine.extract_source_pages(
            [dict(TEXT_CHUNK), second], ["ร้านไอติมโอ่ง", "บ้านจีน 300 ปี"]
        )
        self.assertEqual(pages, [24, 26])
        self.assertEqual(
            SongkhlaLineHandler._format_source_pages(pages),
            "แหล่งข้อมูลอ้างอิง: หน้า 24, 26",
        )

    def test_missing_page_is_not_fabricated(self):
        chunk = dict(TEXT_CHUNK)
        chunk.pop("source_page")
        chunk["metadata"] = {"unrelated": 99}
        pages = SongkhlaRAGEngine.extract_source_pages(
            [chunk], ["ร้านไอติมโอ่ง"]
        )
        text = SongkhlaLineHandler._append_answer_metadata(
            "คำตอบ", {
                "source_pages": pages,
                "provider": "ollama",
                "llm_generated_final_answer": True,
            }
        )
        self.assertNotIn("แหล่งข้อมูลอ้างอิง", text)
        self.assertIn("ประมวลผลโดย: Local LLM", text)

    def test_irrelevant_top_k_pages_are_excluded(self):
        unrelated = [
            {
                "chunk_id": "chunk_027",
                "title": "ร้านเจ๊นิ",
                "source_page": 27,
                "content": "ข้อมูลร้านเจ๊นิ",
            },
            {
                "chunk_id": "chunk_030",
                "title": "บ้านขนมไทย",
                "source_page": 30,
                "content": "ข้อมูลบ้านขนมไทย",
            },
        ]
        engine = make_engine([dict(TEXT_CHUNK), *unrelated])
        engine.call_ollama = lambda messages, model_name: (
            "ร้านไอติมโอ่งเปิดเวลา 10:00 - 18:30 น.ครับ"
        )
        result = engine.generate(
            "ร้านไอติมโอ่งเปิดกี่โมง", target_llm="ollama", user_id=""
        )
        self.assertEqual(result["source_pages"], [24])

    def test_direct_entity_page_wins_over_passing_itinerary_mention(self):
        itinerary = {
            "chunk_id": "chunk_002",
            "title": "โปรแกรมเที่ยวสงขลา 2 วัน",
            "source_page": 5,
            "content": "10.00 น. ขึ้นเขาตังกวน",
        }
        direct = {
            "chunk_id": "chunk_012",
            "title": "เขาตังกวน (ลิฟต์กระเช้าไฟฟ้า)",
            "source_page": 21,
            "content": "ค่าลิฟต์ ผู้ใหญ่ 30 บาท เด็ก 20 บาท",
        }
        pages = SongkhlaRAGEngine.extract_source_pages(
            [itinerary, direct], ["เขาตังกวน"]
        )
        self.assertEqual(pages, [21])

    def test_graph_page_metadata_is_used_when_reliable(self):
        graph = dict(GRAPH_CHUNK)
        graph["metadata"] = {"page_number": "24"}
        pages = SongkhlaRAGEngine.extract_source_pages(
            [graph], ["ร้านไอติมโอ่ง"]
        )
        self.assertEqual(pages, [24])

    def test_follow_up_keeps_referent_and_page(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.sessions["traveler"] = [
            {"role": "user", "content": "ร้านไอติมโอ่งเปิดกี่โมง"},
            {"role": "assistant", "content": "เปิด 10:00 - 18:30 น."},
        ]
        engine.call_ollama = lambda messages, model_name: "ราคา 20 - 30 บาทครับ"
        result = engine.generate(
            "แล้วราคาเท่าไหร่", target_llm="ollama", user_id="traveler"
        )
        self.assertIn("ร้านไอติมโอ่ง", result["retrieval_query"])
        self.assertEqual(result["source_pages"], [24])
        self.assertIn("ประมวลผลโดย: Local LLM", self._line_text(result))

    def test_static_response_has_no_fake_provider(self):
        text = SongkhlaLineHandler._append_answer_metadata(
            "ข้อความต้อนรับ", {
                "provider": "deterministic_static",
                "source_pages": [],
                "llm_generated_final_answer": False,
            }
        )
        self.assertEqual(text, "ข้อความต้อนรับ")

    def test_guardrail_has_no_fabricated_metadata(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda *args: self.fail("LLM must not be called")
        result = engine.generate("ไปบ้านพรุไปยังไง", user_id="")
        text = self._line_text(result)
        self.assertEqual(result["source_pages"], [])
        self.assertNotIn("แหล่งข้อมูลอ้างอิง", text)
        self.assertNotIn("ประมวลผลโดย", text)

    def test_line_split_keeps_footer_once_on_last_part(self):
        answer = "ก" * 9970
        final = SongkhlaLineHandler._append_answer_metadata(
            answer, {
                "source_pages": [24],
                "provider": "ollama",
                "llm_generated_final_answer": True,
            }
        )
        parts = SongkhlaLineHandler._split_line_text(final)
        self.assertTrue(all(0 < len(part) <= 5000 for part in parts))
        self.assertNotIn("แหล่งข้อมูลอ้างอิง", "".join(parts[:-1]))
        self.assertEqual(parts[-1].count("แหล่งข้อมูลอ้างอิง"), 1)
        self.assertEqual(parts[-1].count("ประมวลผลโดย"), 1)


class RoutingObservabilityTests(unittest.TestCase):
    def test_local_route_is_logged_to_terminal(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.call_ollama = lambda messages, model_name: "คำตอบจำลอง"
        output = StringIO()

        with redirect_stdout(output):
            engine.generate(
                "ร้านไอติมโอ่งเปิดกี่โมง",
                target_llm="ollama",
                user_id=""
            )

        terminal = output.getvalue()
        self.assertIn("[LLM ROUTER]", terminal)
        self.assertIn("Provider: LOCAL", terminal)
        self.assertIn("Backend: Ollama", terminal)
        self.assertIn("Model: fake-local", terminal)
        self.assertNotIn("ข้อมูลบริบทอ้างอิง", terminal)

    def test_cloud_failure_logs_local_fallback_without_secrets(self):
        engine = make_engine([dict(TEXT_CHUNK)])
        engine.groq_api_key = "secret-must-not-appear"
        engine.call_groq = lambda messages, model_name: "Cloud API Error: timeout"
        engine.call_ollama = lambda messages, model_name: "คำตอบจาก local"
        output = StringIO()

        with redirect_stdout(output):
            result = engine.generate(
                "ช่วยวางแผนเที่ยวเมืองเก่าสงขลา 2 วัน",
                target_llm="cloud",
                user_id=""
            )

        terminal = output.getvalue()
        self.assertIn("[LLM FALLBACK]", terminal)
        self.assertIn("Original provider: CLOUD/Groq", terminal)
        self.assertIn("Fallback provider: LOCAL/Ollama", terminal)
        self.assertIn("Result: SUCCEEDED", terminal)
        self.assertNotIn("secret-must-not-appear", terminal)
        self.assertEqual(result["provider"], "ollama (failover)")


if __name__ == "__main__":
    unittest.main()
