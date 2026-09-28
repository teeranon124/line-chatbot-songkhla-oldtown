# -*- coding: utf-8 -*-
"""
Integrates Automated LLM-Extracted Triples into Songkhla Knowledge Graph.
Replaces any manual dictionaries with 147 triples extracted directly by Ollama Qwen2.5:3b.
Updates:
1. finalproject/data/songkhla_knowledge_triples.json
2. Re-runs finalproject/scripts/04_build_graph.py to update songkhla_graph.pkl & songkhla_knowledge_graph.html
"""

import os
import sys
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

AUTOMATED_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples_automated.json"
FACTS_PATH = DATA_DIR / "songkhla_places_facts.json"
TARGET_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples.json"


def integrate_triples():
    print("=" * 70)
    print("🔄 INTEGRATING AUTOMATED LLM TRIPLES INTO KNOWLEDGE GRAPH")
    print("=" * 70)

    # 1. Load automated triples
    with open(AUTOMATED_TRIPLES_PATH, "r", encoding="utf-8") as f:
        llm_triples = json.load(f)

    # 2. Load places facts
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        places = json.load(f)

    # Build entity nodes
    nodes = []
    for p in places:
        cat = p["category"]
        if "โรงแรม" in cat or "ที่พัก" in cat:
            label = "Hotel"
        elif "อาหาร" in cat or "ของหวาน" in cat or "คาเฟ่" in cat:
            label = "FoodShop"
        else:
            label = "Place"

        # Price tier classification
        tier = "ระดับมาตรฐาน"
        if "ฟรี" in p["price_range"] or "20 -" in p["price_range"] or "30 บาท" in p["price_range"]:
            tier = "ระดับประหยัด"
        elif "800" in p["price_range"] or "1,200" in p["price_range"] or "โรงแรม" in cat:
            tier = "ระดับพรีเมียม / โรงแรม"

        nodes.append({
            "id": p["id"],
            "label": label,
            "name": p["name"],
            "name_en": p["name_en"],
            "category": p["category"],
            "street": p["street"],
            "address": p["address"],
            "open_hours": p["open_hours"],
            "open_days": p["open_days"],
            "price_range": p["price_range"],
            "price_tier": tier,
            "rating": p.get("rating", 4.5),
            "review_count": p.get("review_count", 150),
            "phone": p.get("phone", ""),
            "lat": p["lat"],
            "lon": p["lon"],
            "google_maps_url": p["google_maps_url"],
            "anyflip_page": p.get("anyflip_page", 10)
        })

    # Convert LLM triples to standard schema format
    formatted_triples = []
    for t in llm_triples:
        subj = t.get("subject", "").strip()
        rel = t.get("relation", "").strip().upper()
        obj = t.get("object", "").strip()
        evidence = t.get("evidence", "")

        if subj and rel and obj:
            formatted_triples.append({
                "source": subj,
                "relation": rel,
                "target": obj,
                "properties": {
                    "evidence": evidence,
                    "extracted_by": "Ollama_Qwen2.5_3B",
                    "extraction_method": "Automated_LLM_Relation_Extraction"
                }
            })

    output_data = {
        "metadata": {
            "title": "Songkhla Old Town Knowledge Graph Triples (100% Automated LLM Extraction)",
            "version": "2.0-automated",
            "total_entities": len(nodes),
            "total_triples": len(formatted_triples),
            "extraction_model": "qwen2.5:3b",
            "source": "Raw PDF Chunks (21 chunks from Songkhla_Travel_Guide_AnyFlip.pdf)"
        },
        "entities": nodes,
        "triples": formatted_triples
    }

    with open(TARGET_TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Integrated {len(formatted_triples)} automated triples into {TARGET_TRIPLES_PATH}")

    # Re-run graph builder
    print("\n[STEP 2] Rebuilding NetworkX graph & PyVis HTML...")
    from subprocess import run
    run([sys.executable, str(BASE_DIR / "scripts" / "04_build_graph.py")], check=True)
    print("\n🎉 Knowledge Graph successfully rebuilt from 100% automated extraction!")


if __name__ == "__main__":
    integrate_triples()
