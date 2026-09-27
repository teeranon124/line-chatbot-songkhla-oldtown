# -*- coding: utf-8 -*-
"""
Flask Webhook and API Server for Songkhla Old Town Assistant.
Features:
- LINE Webhook endpoint (/callback) with signature verification
- Direct REST chat testing endpoint (/chat)
- Intent testing endpoint (/intent)
- Static image server (/static/images/<filename>)
- Interactive Graph visualizer endpoint (/graph)
"""
import sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from flask import Flask, request, jsonify, abort, send_from_directory
from linebot.exceptions import InvalidSignatureError

# Ensure src parent directory is in sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import line_config, paths
from src.rag_engine import SongkhlaRAGEngine
from src.line_handler import SongkhlaLineHandler

app = Flask(__name__)

# Global singletons
rag_engine = None
line_handler = None


def get_services():
    global rag_engine, line_handler
    if rag_engine is None:
        rag_engine = SongkhlaRAGEngine()
        line_handler = SongkhlaLineHandler(rag_engine=rag_engine)
    return rag_engine, line_handler


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online",
        "service": "Songkhla Old Town Hybrid GraphRAG LINE Bot",
        "endpoints": {
            "webhook": "/callback",
            "chat_api": "/chat (POST JSON {'query': '...'})",
            "intent_api": "/intent (POST JSON {'query': '...'})",
            "graph_ui": "/graph (Interactive Knowledge Graph)",
            "health": "/health"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "healthy"})


@app.route("/callback", methods=["POST"])
def callback():
    """LINE Webhook handler."""
    _, handler_service = get_services()
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    print(f"\n📩 [Webhook Received] Signature: {signature[:15]}... Body: {body[:100]}...")

    if not handler_service.handler:
        print("❌ LINE Channel Secret not configured")
        return "LINE Channel Secret not configured", 500

    try:
        handler_service.handler.handle(body, signature)
    except InvalidSignatureError:
        print("❌ [LINE Webhook] Invalid signature received! Check LINE_CHANNEL_SECRET.")
        abort(400)
    except Exception as e:
        print(f"❌ [LINE Webhook Error] {e}")
        import traceback
        traceback.print_exc()
        return "OK", 200

    return "OK", 200


@app.route("/chat", methods=["POST"])
def direct_chat():
    """Direct REST endpoint for testing RAG Engine responses."""
    engine, _ = get_services()
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please provide 'query' in JSON body"}), 400

    mode = data.get("mode", "hybrid")
    llm = data.get("llm", None)
    user_id = data.get("user_id", "test_user")

    result = engine.generate(query=query, mode=mode, target_llm=llm, user_id=user_id)
    return jsonify(result)


@app.route("/intent", methods=["POST"])
def check_intent():
    """Endpoint to inspect intent classification."""
    _, handler_service = get_services()
    data = request.get_json(force=True, silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please provide 'query' in JSON body"}), 400

    res = handler_service.intent_classifier.classify(query)
    return jsonify(res)


@app.route("/graph", methods=["GET"])
def view_graph():
    """Serves the interactive PyVis Knowledge Graph HTML."""
    html_file = paths.data_dir / "songkhla_knowledge_graph.html"
    if html_file.exists():
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}
    return "Graph HTML not generated yet", 404


@app.route("/static/images/<path:filename>", methods=["GET"])
def serve_image(filename):
    """Serves extracted AnyFlip page images and graph diagrams."""
    return send_from_directory(str(paths.static_images_dir), filename)


if __name__ == "__main__":
    get_services()
    port = line_config.port
    print(f"\n🌐 Songkhla Old Town Webhook Server running on http://0.0.0.0:{port}")
    print(f"👉 Forward Cloudflare Tunnel to port {port}/callback for LINE Webhook\n")
    app.run(host="0.0.0.0", port=port, debug=False)
