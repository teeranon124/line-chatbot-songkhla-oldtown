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
