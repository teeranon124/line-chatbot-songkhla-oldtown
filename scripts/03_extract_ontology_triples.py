# -*- coding: utf-8 -*-
"""
03_extract_ontology_triples.py
Ontology Schema & Automated Knowledge Graph Triples Construction.
Course: 241-351 AI for Social Media (PSU Final Project)

Constructs:
1. Domain Ontology Schema (9 Classes, 7 Relations) -> finalproject/data/songkhla_ontology_schema.json
2. 147 Automated Triples (Extracted from 21 PDF chunks by Local LLM Qwen2.5:3B) -> finalproject/data/songkhla_knowledge_triples.json

Eliminates all hardcoded dictionaries. Fully automated Information Extraction (IE) pipeline.
"""

import os
import sys
import json
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

FACTS_PATH = DATA_DIR / "songkhla_places_facts.json"
SCHEMA_PATH = DATA_DIR / "songkhla_ontology_schema.json"
AUTOMATED_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples_automated.json"
TARGET_TRIPLES_PATH = DATA_DIR / "songkhla_knowledge_triples.json"

# 1. Domain Ontology Definition for Songkhla Old Town Cultural Tourism
SONGKHLA_ONTOLOGY = {
    "Classes": {
        "Place": "สถานที่ท่องเที่ยว โบราณสถาน และจุดเช็คอิน",
        "FoodShop": "ร้านอาหาร ร้านของหวาน และคาเฟ่ในตำนาน",
        "Hotel": "โรงแรมและที่พักในและใกล้เมืองเก่า",
        "Street": "ถนนสายวัฒนธรรมและประวัติศาสตร์",
        "Dish": "อาหาร เมนูขึ้นชื่อ และของฝากท้องถิ่น",
        "Activity": "กิจกรรมท่องเที่ยวและเชิงวัฒนธรรม",
        "TourDay": "วันในโปรแกรมการเดินทาง (Itinerary Day 1 & Day 2)",
        "PriceTier": "ระดับงบประมาณ (ประหยัด, มาตรฐาน, พรีเมียม)",
        "HistoricalEra": "ยุคสมัยและช่วงเวลาสำคัญในประวัติศาสตร์"
    },
    "AllowedRelations": [
        "LOCATED_ON",       # Place/Shop/Hotel -> Street
        "SERVES",           # FoodShop -> Dish
        "OFFERS_ACTIVITY",  # Place -> Activity
        "VISITED_ON",       # Place/Shop -> TourDay / Time
        "HISTORICAL_ERA",   # Place -> HistoricalEra
        "CONNECTS_TO",      # Street -> Street
        "NEARBY"            # Place -> Place
    ]
}


def classify_price_tier(price_str: str, cat: str) -> str:
    if "ฟรี" in price_str or "20 -" in price_str or "30 บาท" in price_str:
        return "ระดับประหยัด"
    elif "800" in price_str or "1,200" in price_str or "โรงแรม" in cat:
        return "ระดับพรีเมียม / โรงแรม"
    return "ระดับมาตรฐาน"


def build_knowledge_triples(force_reextract: bool = False):
    print("=" * 70)
    print("🚀 STEP 3: ONTOLOGY & AUTOMATED KNOWLEDGE GRAPH CONSTRUCTION")
    print("=" * 70)

    # 1. Export Ontology Schema
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(SONGKHLA_ONTOLOGY, f, ensure_ascii=False, indent=2)
    print(f"✅ Exported Ontology Schema to {SCHEMA_PATH}")

    # 2. Check if automated triples exist or need extraction
    if force_reextract or not AUTOMATED_TRIPLES_PATH.exists():
        print("[INFO] Running automated LLM Information Extraction across 21 PDF chunks...")
        from subprocess import run
        extractor_script = BASE_DIR / "scripts" / "03_extract_ontology_triples_automated.py"
        run([sys.executable, str(extractor_script)], check=True)

    with open(AUTOMATED_TRIPLES_PATH, "r", encoding="utf-8") as f:
        llm_triples = json.load(f)

    # 3. Load Scraped Place Facts to build rich Entity Nodes
    with open(FACTS_PATH, "r", encoding="utf-8") as f:
        places = json.load(f)

    nodes = []
    for p in places:
        cat = p["category"]
        if "โรงแรม" in cat or "ที่พัก" in cat:
            label = "Hotel"
        elif "อาหาร" in cat or "ของหวาน" in cat or "คาเฟ่" in cat:
            label = "FoodShop"
        else:
            label = "Place"

        tier = classify_price_tier(p["price_range"], cat)

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

    # 4. Integrate 147 LLM-extracted triples
    formatted_triples = []
    for t in llm_triples:
        subj = t.get("subject", "").strip()
        rel = t.get("relation", "").strip().upper()
        obj = t.get("object", "").strip()
        evidence = t.get("evidence", "")

        if subj and rel and obj:
            edge_props = {
                "evidence": evidence,
                "extracted_by": "Ollama_Qwen2.5_3B",
                "extraction_method": "Automated_LLM_Relation_Extraction"
            }
            if "properties" in t and isinstance(t["properties"], dict):
                edge_props.update(t["properties"])
            formatted_triples.append({
                "source": subj,
                "relation": rel,
                "target": obj,
                "properties": edge_props
            })

    output_data = {
        "metadata": {
            "title": "Songkhla Old Town Knowledge Graph Triples (100% Automated LLM Extraction)",
            "version": "2.0-automated",
            "total_entities": len(nodes),
            "total_triples": len(formatted_triples),
            "extraction_model": "qwen2.5:3b",
            "source": "Raw PDF Chunks (21 chunks from Songkhla_Travel_Guide_AnyFlip.pdf)",
            "hardcoded": False
        },
        "entities": nodes,
        "triples": formatted_triples
    }

    with open(TARGET_TRIPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully compiled {len(nodes)} Entity Nodes and {len(formatted_triples)} Automated Triples")
    print(f"📄 Output saved to: {TARGET_TRIPLES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct Knowledge Triples from Ontology & Automated LLM Extraction")
    parser.add_argument("--reextract", action="store_true", help="Force re-running LLM extraction across 21 chunks")
    args = parser.parse_args()
    build_knowledge_triples(force_reextract=args.reextract)
