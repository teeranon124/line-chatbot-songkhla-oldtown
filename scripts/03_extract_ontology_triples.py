# -*- coding: utf-8 -*-
"""
Pipeline for extracting Domain Ontology, Entities, and Knowledge Graph Triples
Combines:
1. Scraped Google Knowledge Facts (songkhla_places_facts.json)
2. AnyFlip Cultural Text & Itinerary Chunks (songkhla_rag_chunks.json)

Outputs:
1. finalproject/data/songkhla_ontology_schema.json
2. finalproject/data/songkhla_knowledge_triples.json
"""

import json
import os
import re

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
        "PriceTier": "ระดับงบประมาณ (ประหยัด, มาตรฐาน, พิเศษ)",
        "HistoricalEra": "ยุคสมัยและช่วงเวลาสำคัญในประวัติศาสตร์"
    },
    "AllowedRelations": [
        "LOCATED_ON",       # Place/Shop/Hotel -> Street
        "SERVES",           # FoodShop -> Dish
        "OFFERS_ACTIVITY",  # Place -> Activity
        "VISITED_ON",       # Place/Shop -> TourDay (มี property: time, order)
        "HAS_PRICE_TIER",   # Place/Shop/Hotel -> PriceTier
        "HISTORICAL_ERA",   # Place -> HistoricalEra
        "NEARBY"            # Place -> Place (มี property: distance_m, walk_min)
    ]
}

ITINERARY_SCHEDULE = [
    # Day 1 Itinerary
    {"place_id": "songkhla_art_center", "day": 1, "order": 1, "time": "09:00", "activity": "ชมนิทรรศการศิลปะร่วมสมัย"},
    {"place_id": "baan_nakorn_in", "day": 1, "order": 2, "time": "10:00", "activity": "ชมพิพิธภัณฑ์บ้านโบราณสองฝั่งถนน"},
    {"place_id": "hub_ho_hin", "day": 1, "order": 3, "time": "11:00", "activity": "ถ่ายภาพอาคารไม้สีแดงและชมวิวทะเลสาบสงขลา"},
    {"place_id": "kiat_fang", "day": 1, "order": 4, "time": "12:00", "activity": "รับประทานข้าวสตูและซาลาเปาลูกใหญ่"},
    {"place_id": "baan_chinese_300yr", "day": 1, "order": 5, "time": "13:00", "activity": "ชมสถาปัตยกรรมเรือนไม้จีนโบราณ"},
    {"place_id": "hotel_songkhla_taeraek", "day": 1, "order": 6, "time": "14:00", "activity": "เช็คอินเข้าที่พักสไตล์แอนทีค"},
    {"place_id": "baan_ww2", "day": 1, "order": 7, "time": "15:00", "activity": "ชมร่องรอยประวัติศาสตร์สงครามมหาเอเชียบูรพา"},
    {"place_id": "city_pillar_shrine", "day": 1, "order": 8, "time": "16:00", "activity": "สักการะศาลเจ้าพ่อหลักเมืองสงขลา"},
    {"place_id": "songkhla_street_art", "day": 1, "order": 9, "time": "17:00", "activity": "เดินถ่ายรูปสตรีทอาร์ทบนกำแพงเมืองเก่า"},
    {"place_id": "aitim_oang", "day": 1, "order": 10, "time": "17:30", "activity": "ชิมไอติมโอ่งโบราณและไอติมไข่แข็ง"},
    {"place_id": "tae_hiang_iu", "day": 1, "order": 11, "time": "18:00", "activity": "รับประทานอาหารค่ำ ต้มยำแห้งปลากระพง"},
    
    # Day 2 Itinerary
    {"place_id": "singora_tram", "day": 2, "order": 1, "time": "08:00", "activity": "นั่งรถรางชมเมืองสงขลา แหลมสมิหลา นางเงือกทอง"},
    {"place_id": "khao_tang_kuan", "day": 2, "order": 2, "time": "10:00", "activity": "ขึ้นลิฟต์กระเช้าไฟฟ้าชมวิว 360 องศา"},
    {"place_id": "songkhla_station", "day": 2, "order": 3, "time": "11:00", "activity": "จิบกาแฟสดและชมภาพถ่ายเมืองเก่า"},
    {"place_id": "jae_ni", "day": 2, "order": 4, "time": "12:00", "activity": "รับประทานข้าวต้มปลากะพงและหมี่ซั่วแห้ง"},
    {"place_id": "khanom_thai_song_saen", "day": 2, "order": 5, "time": "13:00", "activity": "ซื้อของฝากขนมทองเอกและสัมปันนี"}
]

HISTORICAL_ERA_MAPPING = {
    "hub_ho_hin": "สมัยรัชกาลที่ 6 (พ.ศ. 2457)",
    "kiat_fang": "ยุคก่อนสงครามโลก (พ.ศ. 2480)",
    "baan_chinese_300yr": "ยุคการค้าทางทะเลจีนโบราณ (กว่า 300 ปี)",
    "baan_ww2": "สงครามมหาเอเชียบูรพา (พ.ศ. 2484)",
    "city_pillar_shrine": "สมัยรัชกาลที่ 3 (การสร้างเมืองสงขลาฝั่งบ่อยาง)",
    "khao_tang_kuan": "สมัยพระบาทสมเด็จพระจอมเกล้าเจ้าอยู่หัว (ร.4)",
    "khanom_thai_song_saen": "ยุคหลังสงคราม (พ.ศ. 2490)"
}

PROXIMITY_EDGES = [
    {"from": "hotel_songkhla_taeraek", "to": "songkhla_street_art", "distance_m": 300, "walk_min": 4},
    {"from": "hotel_songkhla_taeraek", "to": "hub_ho_hin", "distance_m": 450, "walk_min": 6},
    {"from": "hotel_songkhla_taeraek", "to": "city_pillar_shrine", "distance_m": 250, "walk_min": 3},
    {"from": "aitim_oang", "to": "city_pillar_shrine", "distance_m": 20, "walk_min": 1},
    {"from": "tae_hiang_iu", "to": "khanom_thai_song_saen", "distance_m": 15, "walk_min": 1},
    {"from": "songkhla_station", "to": "hub_ho_hin", "distance_m": 40, "walk_min": 1},
    {"from": "jae_ni", "to": "hub_ho_hin", "distance_m": 30, "walk_min": 1},
    {"from": "baan_nakorn_in", "to": "hub_ho_hin", "distance_m": 120, "walk_min": 2}
]


def classify_price_tier(price_str):
    if "ฟรี" in price_str or "20 -" in price_str or "30 บาท" in price_str:
        return "ระดับประหยัด (Budget < 50 บาท)"
    elif "60 -" in price_str or "120" in price_str or "150" in price_str or "100" in price_str:
        return "ระดับมาตรฐาน (Standard 50 - 150 บาท)"
    else:
        return "ระดับพรีเมียม / โรงแรม (Premium > 150 บาท)"


def build_knowledge_triples():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    facts_path = os.path.join(base_dir, "songkhla_places_facts.json")
    
    with open(facts_path, "r", encoding="utf-8") as f:
        places = json.load(f)

    nodes = []
    triples = []
    
    # 1. Create Core Place/Shop/Hotel Nodes (Enriched with Scraped Facts)
    for p in places:
        # Determine ontology class
        cat = p["category"]
        if "โรงแรม" in cat or "ที่พัก" in cat:
            node_label = "Hotel"
        elif "อาหาร" in cat or "ของหวาน" in cat or "คาเฟ่" in cat:
            node_label = "FoodShop"
        else:
            node_label = "Place"

        # Price tier classification
        tier = classify_price_tier(p["price_range"])

        node = {
            "id": p["id"],
            "label": node_label,
            "name": p["name"],
            "name_en": p["name_en"],
            "category": p["category"],
            "street": p["street"],
            "address": p["address"],
            "open_hours": p["open_hours"],
            "open_days": p["open_days"],
            "price_range": p["price_range"],
            "price_tier": tier,
            "rating": p["rating"],
            "review_count": p["review_count"],
            "phone": p["phone"],
            "landmark_clue": p["landmark_clue"],
            "lat": p["lat"],
            "lon": p["lon"],
            "google_maps_url": p["google_maps_url"],
            "anyflip_page": p["anyflip_page"]
        }
        nodes.append(node)

        # Relation 1: (:Place)-[:LOCATED_ON]->(:Street)
        triples.append({
            "source": p["name"],
            "source_id": p["id"],
            "source_type": node_label,
            "relation": "LOCATED_ON",
            "target": p["street"],
            "target_type": "Street",
            "properties": {}
        })

        # Relation 2: (:Place)-[:HAS_PRICE_TIER]->(:PriceTier)
        triples.append({
            "source": p["name"],
            "source_id": p["id"],
            "source_type": node_label,
            "relation": "HAS_PRICE_TIER",
            "target": tier,
            "target_type": "PriceTier",
            "properties": {"price_range": p["price_range"]}
        })

        # Relation 3: (:FoodShop)-[:SERVES]->(:Dish)
        if "signature_items" in p:
            for dish in p["signature_items"]:
                # Check if food item or activity
                is_dish = node_label == "FoodShop"
                rel_name = "SERVES" if is_dish else "OFFERS_ACTIVITY"
                target_type = "Dish" if is_dish else "Activity"
                
                triples.append({
                    "source": p["name"],
                    "source_id": p["id"],
                    "source_type": node_label,
                    "relation": rel_name,
                    "target": dish,
                    "target_type": target_type,
                    "properties": {}
                })

        # Relation 4: (:Place)-[:HISTORICAL_ERA]->(:HistoricalEra)
        if p["id"] in HISTORICAL_ERA_MAPPING:
            era = HISTORICAL_ERA_MAPPING[p["id"]]
            triples.append({
                "source": p["name"],
                "source_id": p["id"],
                "source_type": node_label,
                "relation": "HISTORICAL_ERA",
                "target": era,
                "target_type": "HistoricalEra",
                "properties": {}
            })

    # 2. Add Itinerary Triples: (:Place)-[:VISITED_ON]->(:TourDay)
    place_map = {p["id"]: p["name"] for p in places}
    for item in ITINERARY_SCHEDULE:
        pid = item["place_id"]
        pname = place_map.get(pid, pid)
        day_str = f"โปรแกรมเที่ยวสงขลา วันที่ {item['day']}"
        
        triples.append({
            "source": pname,
            "source_id": pid,
            "source_type": "Place",
            "relation": "VISITED_ON",
            "target": day_str,
            "target_type": "TourDay",
            "properties": {
                "day": item["day"],
                "order": item["order"],
                "scheduled_time": item["time"],
                "itinerary_activity": item["activity"]
            }
        })

    # 3. Add Proximity Triples: (:Place)-[:NEARBY]->(:Place)
    for prox in PROXIMITY_EDGES:
        s_name = place_map.get(prox["from"], prox["from"])
        t_name = place_map.get(prox["to"], prox["to"])
        
        triples.append({
            "source": s_name,
            "source_id": prox["from"],
            "source_type": "Place",
            "relation": "NEARBY",
            "target": t_name,
            "target_type": "Place",
            "properties": {
                "distance_m": prox["distance_m"],
                "walk_min": prox["walk_min"]
            }
        })

    # Export Schema JSON
    schema_path = os.path.join(base_dir, "songkhla_ontology_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(SONGKHLA_ONTOLOGY, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported Ontology Schema to {schema_path}")

    # Export Knowledge Triples JSON
    output_data = {
        "metadata": {
            "title": "Songkhla Old Town Knowledge Graph Triples",
            "version": "1.0",
            "total_entities": len(nodes),
            "total_triples": len(triples),
            "sources": [
                "Songkhla_Travel_Guide_AnyFlip.pdf (21 Selected Pages)",
                "Google Knowledge Panel / Wongnai Verified Facts (18 Places)"
            ]
        },
        "entities": nodes,
        "triples": triples
    }
    
    triples_path = os.path.join(base_dir, "songkhla_knowledge_triples.json")
    with open(triples_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"[OK] Exported {len(triples)} Triples and {len(nodes)} Entities to {triples_path}")


if __name__ == "__main__":
    build_knowledge_triples()
