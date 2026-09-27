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
        Extracts subgraph matching entities in the query and returns formatted context chunks.
        """
        if not self.G or self.G.number_of_nodes() == 0:
            return []

        matched_nodes = []
        q_lower = query.lower()

        # 1. Node name and alias matching
        for n, data in self.G.nodes(data=True):
            n_str = str(n).lower()
            name_en = str(data.get("name_en", "")).lower()
            cat = str(data.get("category", "")).lower()
            
            # Exact or partial match
            if n_str in q_lower or any(word in q_lower for word in n_str.split() if len(word) > 2):
                matched_nodes.append((n, data, 2.0))
            elif name_en and name_en in q_lower:
                matched_nodes.append((n, data, 1.5))
            elif cat and cat in q_lower:
                matched_nodes.append((n, data, 1.0))

        # Sort by relevance
        matched_nodes.sort(key=lambda x: x[2], reverse=True)
        top_nodes = matched_nodes[:top_k]

        results = []
        for n, data, score in top_nodes:
            # Traversal 1-hop outgoing edges
            relations_out = []
            for _, target, edge_data in self.G.out_edges(n, data=True):
                rel = edge_data.get("relation", "RELATED_TO")
                props_str = ""
                if rel == "VISITED_ON":
                    props_str = f" (เวลา {edge_data.get('scheduled_time', '')}, ลำดับที่ {edge_data.get('order', '')})"
                elif rel == "NEARBY":
                    props_str = f" (ระยะทาง {edge_data.get('distance_m', '')} ม., เดิน {edge_data.get('walk_min', '')} นาที)"
                relations_out.append(f"{rel} -> {target}{props_str}")

            # Traversal 1-hop incoming edges
            relations_in = []
            for source, _, edge_data in self.G.in_edges(n, data=True):
                rel = edge_data.get("relation", "RELATED_TO")
                relations_in.append(f"{source} -[{rel}]-> {n}")

            content_text = (
                f"ความสัมพันธ์ใน Knowledge Graph สำหรับ: {n}\n"
                f"- ประเภท: {data.get('label', 'Concept')} ({data.get('category', '')})\n"
                f"- ถนน: {data.get('street', 'ย่านเมืองเก่าสงขลา')}\n"
                f"- เวลาเปิด-ปิด: {data.get('open_hours', 'ไม่ระบุ')}\n"
                f"- ช่วงราคา: {data.get('price_range', 'ไม่ระบุ')}\n"
                f"- คะแนนรีวิว: {data.get('rating', 'ไม่มีข้อมูล')} ดาว ({data.get('review_count', 0)} รีวิว)\n"
                f"- โครงสร้างความสัมพันธ์ที่เชื่อมโยง: {'; '.join(relations_out[:6])}"
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
        """Retrieves chronological itinerary stops for Day 1 or Day 2."""
        items = []
        for u, v, d in self.G.edges(data=True):
            if d.get("relation") == "VISITED_ON":
                item_day = d.get("day", 1)
                if day is None or item_day == day:
                    items.append({
                        "place": u,
                        "day": item_day,
                        "order": d.get("order", 99),
                        "time": d.get("scheduled_time", ""),
                        "activity": d.get("itinerary_activity", ""),
                        "node_data": self.G.nodes[u]
                    })
        items.sort(key=lambda x: (x["day"], x["order"]))
        return items

    def get_nearby_places(self, place_name: str) -> List[Dict[str, Any]]:
        """Returns places near the given place with walking distances."""
        nearby = []
        for u, v, d in self.G.edges(data=True):
            if place_name.lower() in str(u).lower() and d.get("relation") == "NEARBY":
                nearby.append({
                    "from": u,
                    "to": v,
                    "distance_m": d.get("distance_m", 0),
                    "walk_min": d.get("walk_min", 0),
                    "target_data": self.G.nodes.get(v, {})
                })
        nearby.sort(key=lambda x: x["distance_m"])
        return nearby
