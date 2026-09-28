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

        matched_nodes = []
        q_lower = query.lower()

        # 1. Node name and alias matching with score prioritization
        for n, data in self.G.nodes(data=True):
            n_str = str(n).lower()
            name_en = str(data.get("name_en", "")).lower()
            cat = str(data.get("category", "")).lower()
            street = str(data.get("street", "")).lower()
            
            # Exact match gets highest priority
            if n_str == q_lower:
                matched_nodes.append((n, data, 3.0))
            elif n_str in q_lower:
                matched_nodes.append((n, data, 2.5))
            elif any(word in q_lower for word in n_str.split() if len(word) > 2):
                matched_nodes.append((n, data, 2.0))
            elif name_en and name_en in q_lower:
                matched_nodes.append((n, data, 1.8))
            elif street and street in q_lower:
                matched_nodes.append((n, data, 1.5))
            elif cat and cat in q_lower:
                matched_nodes.append((n, data, 1.0))

        # Deduplicate and sort by relevance score
        seen = set()
        unique_matched = []
        for n, data, score in sorted(matched_nodes, key=lambda x: x[2], reverse=True):
            if n not in seen:
                seen.add(n)
                unique_matched.append((n, data, score))

        top_nodes = unique_matched[:top_k]

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
