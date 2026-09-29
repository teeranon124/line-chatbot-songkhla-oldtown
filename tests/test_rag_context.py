# -*- coding: utf-8 -*-
import sys
import types
import unittest
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

    def retrieve(self, query, top_k=None, mode="hybrid"):
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


if __name__ == "__main__":
    unittest.main()
