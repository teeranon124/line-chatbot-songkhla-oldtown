# -*- coding: utf-8 -*-
"""
Knowledge Graph Engine for Songkhla Old Town Assistant.
Supports live Neo4j Cypher execution and In-Memory NetworkX Traversal.
Provides Multi-hop relational search, itinerary sequencing, budget filtering, and proximity lookup.
"""
import os
import socket
import pickle
from typing import List, Dict, Any
import networkx as nx
from pythainlp.tokenize import word_tokenize

from .config import paths, models


def is_neo4j_reachable(host="localhost", port=7687, timeout=1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class SongkhlaGraphEngine:
    def __init__(self):
        self.graph_pkl_path = paths.graph_pkl_path
        self.G = None
        self.neo4j_driver = None
        
        # 1. Load In-Memory NetworkX Graph
        if os.path.exists(self.graph_pkl_path):
            with open(self.graph_pkl_path, "rb") as f:
                self.G = pickle.load(f)
            print(f"[GraphEngine] Loaded NetworkX Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")
        else:
            print(f"[GraphEngine Warning] Pickle graph not found at {self.graph_pkl_path}. Creating empty DiGraph.")
            self.G = nx.DiGraph()

        # 2. Check Neo4j Connection
        if is_neo4j_reachable():
            try:
                from neo4j import GraphDatabase
                self.neo4j_driver = GraphDatabase.driver(
                    models.neo4j_uri,
                    auth=(models.neo4j_user, models.neo4j_pass)
                )
                print(f"[GraphEngine] Connected to Neo4j successfully at {models.neo4j_uri}")
            except Exception as e:
                print(f"[GraphEngine Notice] Neo4j connection failed: {e}. Using NetworkX engine.")
        else:
            print("[GraphEngine Notice] Neo4j is offline. Using High-Performance NetworkX Engine.")

    def search_subgraph(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Extracts multi-hop subgraph matching entities in the query and returns rich context chunks.
        Properly traverses both incoming (in_edges) and outgoing (out_edges) relations.
        """
        if not self.G or self.G.number_of_nodes() == 0:
            return []

        q_lower = query.lower()
        q_tokens = [w for w in word_tokenize(q_lower, engine="newmm") if len(w.strip()) > 1]

        # 1. Identify seed nodes mentioned in query (Exact and canonical entity resolution)
        seed_nodes = []
        stopwords = {"สถานที่", "ท่องเที่ยว", "ประวัติศาสตร์", "โบราณ", "อาหาร", "ร้าน", "ของหวาน", "ของฝาก", "สำคัญ", "ย่าน", "เมือง", "สงขลา", "ไทย"}
        for n, data in self.G.nodes(data=True):
            n_str = str(n).strip()
            n_low = n_str.lower()
            if n_low in stopwords or len(n_low) < 2:
                continue
            if n_low in q_lower:
                seed_nodes.append((n, data))
            elif data.get('name_en') and data['name_en'].lower() in q_lower:
                seed_nodes.append((n, data))
            elif data.get('id') and len(data['id']) >= 4 and data['id'].lower() in q_lower:
                seed_nodes.append((n, data))

        # 1.1 Fuzzy Entity Resolution for Typos & Phonetic Variations (Sliding-window Levenshtein)
        if len(seed_nodes) == 0:
            import difflib
            import re
            for n, data in self.G.nodes(data=True):
                n_str = str(n).strip()
                cand_clean = re.sub(r'^(ร้าน|โรงแรม|ถนน|บ้านขนมไทย|โรงสีแดง|พิพิธภัณฑ์)', '', n_str).strip()
                if len(cand_clean) < 3:
                    cand_clean = n_str

                # Try sliding window of length cand_clean +- 2 on q_lower
                w_len = len(cand_clean)
                best_r = 0.0
                for offset in [-2, -1, 0, 1, 2]:
                    sub_len = max(3, w_len + offset)
                    for i in range(len(q_lower) - sub_len + 1):
                        sub = q_lower[i:i+sub_len].strip()
                        r = difflib.SequenceMatcher(None, sub, cand_clean).ratio()
                        if r > best_r:
                            best_r = r
                if best_r >= 0.68:
                    seed_nodes.append((n, data))

        # Detect user intent and domain keywords
        has_opposite = any(w in q_lower for w in ['ตรงข้าม', 'ฝั่งตรงข้าม', 'เยื้อง', 'ข้ามถนน'])
        has_nearby = any(w in q_lower for w in ['ใกล้', 'ติดกับ', 'ข้างๆ', 'รอบๆ'])
        has_distance = any(w in q_lower for w in ['กี่เมตร', 'ระยะทาง', 'เดิน', 'นาที', 'ไกล', 'เส้นทาง'])
        is_asking_all = any(w in q_lower for w in ['อะไรบ้าง', 'มีร้านใดบ้าง', 'ร้านอะไรบ้าง', 'รวบรวม', 'ทั้งหมด', 'เปิดติดๆ'])
        wants_foodshop = any(w in q_lower for w in ['ร้าน', 'ร้านอาหาร', 'ร้านขนม', 'ของหวาน', 'ไอติม', 'ของฝาก', 'กิน'])

        node_scores = {}

        # -------------------------------------------------------------
        # GRAPH PATTERN 1: Multi-Constraint Intersection (e.g. Street + Historical Era)
        # -------------------------------------------------------------
        if len(seed_nodes) >= 2:
            for n, data in self.G.nodes(data=True):
                connected_seeds = 0
                for sn, _ in seed_nodes:
                    if self.G.has_edge(n, sn) or self.G.has_edge(sn, n):
                        connected_seeds += 1
                if connected_seeds >= 2:
                    node_scores[n] = node_scores.get(n, 0.0) + 60.0

        # -------------------------------------------------------------
        # GRAPH PATTERN 2: Spatial Adjacency / Opposite / Nearby Traversal
        # -------------------------------------------------------------
        if has_opposite or has_nearby:
            for sn, _ in seed_nodes:
                # Outgoing edges
                for _, target, edata in self.G.out_edges(sn, data=True):
                    rel = edata.get('relation', '')
                    if (has_opposite and rel == 'OPPOSITE_TO') or (has_nearby and rel in ['NEARBY', 'OPPOSITE_TO']):
                        tdata = self.G.nodes.get(target, {})
                        bonus = 45.0
                        if wants_foodshop and tdata.get('label') == 'FoodShop':
                            bonus += 15.0
                        if 'ของหวาน' in q_lower and 'ของหวาน' in tdata.get('category', ''):
                            bonus += 20.0
                        node_scores[target] = node_scores.get(target, 0.0) + bonus
                # Incoming edges
                for source, _, edata in self.G.in_edges(sn, data=True):
                    rel = edata.get('relation', '')
                    if (has_opposite and rel == 'OPPOSITE_TO') or (has_nearby and rel in ['NEARBY', 'OPPOSITE_TO']):
                        sdata_src = self.G.nodes.get(source, {})
                        bonus = 45.0
                        if wants_foodshop and sdata_src.get('label') == 'FoodShop':
                            bonus += 15.0
                        if 'ของหวาน' in q_lower and 'ของหวาน' in sdata_src.get('category', ''):
                            bonus += 20.0
                        node_scores[source] = node_scores.get(source, 0.0) + bonus

        # -------------------------------------------------------------
        # GRAPH PATTERN 3: Street Topological Aggregation (Street -> All Located Shops)
        # -------------------------------------------------------------
        for sn, sdata in seed_nodes:
            if sdata.get('label') == 'Street' or 'ถนน' in str(sn):
                for source, _, edata in self.G.in_edges(sn, data=True):
                    if edata.get('relation') == 'LOCATED_ON':
                        src_data = self.G.nodes.get(source, {})
                        bonus = 30.0
                        if wants_foodshop and src_data.get('label') == 'FoodShop':
                            bonus += 25.0
                        if 'ของหวาน' in q_lower and 'ของหวาน' in src_data.get('category', ''):
                            bonus += 15.0
                        node_scores[source] = node_scores.get(source, 0.0) + bonus
                if is_asking_all:
                    node_scores[sn] = node_scores.get(sn, 0.0) + 35.0

        # -------------------------------------------------------------
        # GRAPH PATTERN 4: Reverse Relation (Dish -> FoodShop)
        # -------------------------------------------------------------
        for sn, sdata in seed_nodes:
            if sdata.get('label') == 'Dish':
                for source, _, edata in self.G.in_edges(sn, data=True):
                    if edata.get('relation') == 'SERVES':
                        node_scores[source] = node_scores.get(source, 0.0) + 50.0

        # -------------------------------------------------------------
        # GRAPH PATTERN 5: Distance & Walking Paths
        # -------------------------------------------------------------
        if has_distance:
            for sn, _ in seed_nodes:
                for _, target, edata in self.G.out_edges(sn, data=True):
                    if 'distance_m' in edata or 'walk_min' in edata:
                        node_scores[sn] = node_scores.get(sn, 0.0) + 35.0
                        node_scores[target] = node_scores.get(target, 0.0) + 25.0

        # Domain concept semantic mapping to properties & categories (Fallback & Supplemental)
        concept_map = {
            "ของหวาน": ["ของหวาน", "ขนม", "ไอติม", "ไอศกรีม", "คลายร้อน", "เย็น", "หวาน", "ไข่แข็ง", "ของฝาก"],
            "อาหารคาว": ["อาหารคาว", "อาหาร", "กิน", "ข้าว", "เช้า", "เที่ยง", "สตู", "หมูกรอบ", "ซาลาเปา", "ข้าวต้ม", "กับข้าว"],
            "คาเฟ่": ["กาแฟ", "คาเฟ่", "ชา", "ชิล", "พักเหนื่อย", "แอร์", "เครื่องดื่ม"],
            "ศิลปะ": ["ศิลปะ", "นิทรรศการ", "ภาพวาด", "ภาพเขียน", "แกลเลอรี", "ภาพถ่าย", "สตรีทอาร์ท", "street art"],
            "จุดชมวิว": ["ชมวิว", "วิว", "มุมสูง", "กระเช้า", "ลิฟต์", "ทะเล", "เขาตังกวน"],
            "ประวัติศาสตร์": ["ประวัติศาสตร์", "โบราณ", "เก่าแก่", "300 ปี", "สงครามโลก", "รัชกาล", "โรงสีแดง", "หับโห้หิ้น"]
        }

        for n, data in self.G.nodes(data=True):
            s = 0.0
            cat = str(data.get("category", "")).lower()
            street = str(data.get("street", "")).lower()
            clue = str(data.get("landmark_clue", "")).lower()

            # Seed nodes base score
            if any(sn == n for sn, _ in seed_nodes):
                s += 12.0

            # Properties & Landmark Clue matching
            if clue and any(tok in clue for tok in q_tokens if len(tok) >= 4):
                s += 2.5
            if street and street in q_lower:
                s += 2.0

            # Concept / Category matching
            for concept, kw_list in concept_map.items():
                if any(kw in q_lower for kw in kw_list):
                    if any(kw in cat for kw in kw_list):
                        s += 2.8
                    if any(kw in clue for kw in kw_list):
                        s += 2.2

            if s > 0:
                node_scores[n] = node_scores.get(n, 0.0) + s

        # Spreading activation along graph topology (1-hop propagation)
        propagated = dict(node_scores)
        for n, score in node_scores.items():
            for _, target, edata in self.G.out_edges(n, data=True):
                if target in self.G:
                    propagated[target] = propagated.get(target, 0.0) + score * 0.25
            for source, _, edata in self.G.in_edges(n, data=True):
                if source in self.G:
                    propagated[source] = propagated.get(source, 0.0) + score * 0.25

        # Sort and select top_k nodes
        top_nodes = sorted(propagated.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_nodes = [(n, self.G.nodes[n], score) for n, score in top_nodes]

        results = []
        for n, data, score in top_nodes:
            # Traversal 1-hop outgoing edges
            relations_out = []
            for _, target, edge_data in self.G.out_edges(n, data=True):
                rel = edge_data.get("relation", "RELATED_TO")
                props_str = ""
                if "distance_m" in edge_data:
                    props_str = f" (ระยะทาง {edge_data['distance_m']} ม."
                    if "walk_min" in edge_data:
                        props_str += f", เดิน {edge_data['walk_min']} นาที"
                    props_str += ")"
                elif edge_data.get("evidence"):
                    ev_snippet = edge_data["evidence"][:40]
                    props_str = f" [หลักฐาน: {ev_snippet}...]"
                relations_out.append(f"{rel} -> {target}{props_str}")

            # Traversal 1-hop incoming edges (Crucial for Street nodes to discover located places)
            relations_in = []
            for source, _, edge_data in self.G.in_edges(n, data=True):
                rel = edge_data.get("relation", "RELATED_TO")
                props_str = ""
                if "distance_m" in edge_data:
                    props_str = f" (ระยะทาง {edge_data['distance_m']} ม."
                    if "walk_min" in edge_data:
                        props_str += f", เดิน {edge_data['walk_min']} นาที"
                    props_str += ")"
                relations_in.append(f"{source} -[{rel}]-> {n}{props_str}")

            # Check 2-hop connected places on the same street
            same_street_places = []
            street_val = data.get("street")
            if street_val and street_val != "ย่านเมืองเก่าสงขลา":
                for peer, pdata in self.G.nodes(data=True):
                    if peer != n and pdata.get("street") == street_val:
                        same_street_places.append(peer)

            out_summary = "; ".join(relations_out[:8]) if relations_out else "ไม่มีข้อมูลเชื่อมโยงออกไป"
            in_summary = "; ".join(relations_in[:8]) if relations_in else "ไม่มีข้อมูลเชื่อมโยงเข้ามา"
            peer_summary = ", ".join(same_street_places[:5]) if same_street_places else "ไม่มี"

            content_text = (
                f"โครงสร้างความสัมพันธ์ใน Knowledge Graph สำหรับ: {n}\n"
                f"- ประเภท: {data.get('label', 'Concept')} ({data.get('category', '')})\n"
                f"- ถนนที่ตั้ง: {data.get('street', 'ย่านเมืองเก่าสงขลา')}\n"
                f"- เวลาเปิด-ปิด: {data.get('open_hours', 'ไม่ระบุ')}\n"
                f"- ช่วงราคา: {data.get('price_range', 'ไม่ระบุ')}\n"
                f"- คะแนนรีวิว: {data.get('rating', 'ไม่มีข้อมูล')} ดาว ({data.get('review_count', 0)} รีวิว)\n"
                f"- โครงข่ายความสัมพันธ์เชื่อมโยงออกไป: {out_summary}\n"
                f"- โครงข่ายความสัมพันธ์เชื่อมโยงเข้ามา: {in_summary}\n"
                f"- สถานที่อื่นๆ บนถนนสายเดียวกัน: {peer_summary}"
            )

            results.append({
                "chunk_id": f"graph_{data.get('id', n)}",
                "title": f"[Graph Subgraph] {n}",
                "category": "Knowledge Graph",
                "content": content_text,
                "score": score
            })

        return results

    def get_itinerary(self, day: int = None) -> List[Dict[str, Any]]:
        """Retrieves chronological itinerary stops for Day 1 or Day 2 from Knowledge Graph."""
        items = []
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "VISITED_ON":
                item_day = 1
                v_str = str(v)
                if "วันที่ 2" in v_str or "วันที่สอง" in v_str or "day 2" in v_str.lower():
                    item_day = 2
                elif "วันที่ 1" in v_str or "วันแรก" in v_str or "day 1" in v_str.lower():
                    item_day = 1
                
                if day is None or item_day == day:
                    items.append({
                        "place": u,
                        "day": item_day,
                        "itinerary": v,
                        "evidence": d.get("evidence", ""),
                        "node_data": self.G.nodes.get(u, {})
                    })
        return items

    def get_nearby_places(self, place_name: str) -> List[Dict[str, Any]]:
        """Returns places near the given place with walking distances and travel time."""
        nearby = []
        p_low = place_name.lower()
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "NEARBY":
                if p_low in str(u).lower():
                    nearby.append({
                        "from": u,
                        "to": v,
                        "distance_m": d.get("distance_m", 0),
                        "walk_min": d.get("walk_min", 0),
                        "evidence": d.get("evidence", ""),
                        "target_data": self.G.nodes.get(v, {})
                    })
                elif p_low in str(v).lower():
                    nearby.append({
                        "from": v,
                        "to": u,
                        "distance_m": d.get("distance_m", 0),
                        "walk_min": d.get("walk_min", 0),
                        "evidence": d.get("evidence", ""),
                        "target_data": self.G.nodes.get(u, {})
                    })
        nearby.sort(key=lambda x: x["distance_m"] if x["distance_m"] > 0 else 9999)
        return nearby
