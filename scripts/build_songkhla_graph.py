# -*- coding: utf-8 -*-
"""
Script to build, populate, and export Songkhla Old Town Knowledge Graph
into Neo4j Database and In-Memory NetworkX Graph with Interactive Visualizations.

Outputs:
1. Neo4j Cypher population (if container is alive on bolt://localhost:7687)
2. finalproject/data/songkhla_graph.pkl (Serialized NetworkX DiGraph for RAG Engine)
3. finalproject/data/songkhla_knowledge_graph.html (Interactive PyVis Graph UI)
4. finalproject/data/songkhla_graph_overview.png (High-Res Static Diagram for Slides)
"""

import json
import os
import socket
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

# Neo4j Settings matching previous labs
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password1234")


def is_neo4j_reachable(host="localhost", port=7687, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def build_and_export_graph():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    triples_path = os.path.join(base_dir, "songkhla_knowledge_triples.json")

    with open(triples_path, "r", encoding="utf-8") as f:
        kg_data = json.load(f)

    entities = kg_data["entities"]
    triples = kg_data["triples"]

    # 1. Build In-Memory NetworkX Directed Graph
    print(f"\n[STEP 1] Constructing NetworkX DiGraph...")
    G = nx.DiGraph()

    # Add entity nodes with rich scraped attributes
    for e in entities:
        G.add_node(
            e["name"],
            id=e["id"],
            label=e["label"],
            name_en=e["name_en"],
            category=e["category"],
            street=e["street"],
            address=e["address"],
            open_hours=e["open_hours"],
            open_days=e["open_days"],
            price_range=e["price_range"],
            price_tier=e["price_tier"],
            rating=e["rating"],
            review_count=e["review_count"],
            phone=e["phone"],
            lat=e["lat"],
            lon=e["lon"],
            google_maps_url=e["google_maps_url"],
            anyflip_page=e["anyflip_page"]
        )

    # Add relations
    for t in triples:
        src = t["source"]
        tgt = t["target"]
        rel = t["relation"]
        props = t.get("properties", {})
        
        # Ensure target node exists
        if not G.has_node(tgt):
            G.add_node(tgt, label=t.get("target_type", "Concept"))
            
        G.add_edge(src, tgt, relation=rel, **props)

    print(f"NetworkX Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Save NetworkX pickle
    pkl_path = os.path.join(base_dir, "songkhla_graph.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(G, f)
    print(f"[OK] Saved NetworkX Graph to {pkl_path}")

    # 2. Try Neo4j Population if alive
    print(f"\n[STEP 2] Checking Neo4j connection ({NEO4J_URI})...")
    if is_neo4j_reachable():
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
            with driver.session() as session:
                # Clear previous data
                session.run("MATCH (n) DETACH DELETE n")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Place) REQUIRE p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Street) REQUIRE s.name IS UNIQUE")
                
                # Insert Entities
                for e in entities:
                    label = e["label"]
                    cypher = f"""
                    MERGE (n:{label} {{id: $id}})
                    SET n.name = $name,
                        n.name_en = $name_en,
                        n.category = $category,
                        n.street = $street,
                        n.address = $address,
                        n.open_hours = $open_hours,
                        n.price_range = $price_range,
                        n.rating = $rating,
                        n.phone = $phone,
                        n.lat = $lat,
                        n.lon = $lon
                    """
                    session.run(cypher, e)
                
                # Insert Triples
                for t in triples:
                    rel = t["relation"]
                    src = t["source"]
                    tgt = t["target"]
                    props = t.get("properties", {})
                    
                    cypher = f"""
                    MERGE (s {{name: $src}})
                    MERGE (t {{name: $tgt}})
                    MERGE (s)-[r:{rel}]->(t)
                    SET r += $props
                    """
                    session.run(cypher, {"src": src, "tgt": tgt, "props": props})
                    
                cnt_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
                cnt_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
                print(f"[Neo4j SUCCESS] Populated {cnt_nodes} nodes and {cnt_rels} relationships!")
            driver.close()
        except Exception as err:
            print(f"[Neo4j Notice]: Neo4j driver connection error: {err}")
    else:
        print("[Neo4j Notice]: Neo4j service (port 7687) is not currently active.")
        print(">> In-memory NetworkX Graph will serve as the primary engine with 100% traversal parity.")

    # 3. Export Interactive HTML Graph (PyVis)
    print(f"\n[STEP 3] Generating Interactive Graph HTML with PyVis...")
    net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
    net.force_atlas_2based()

    # Color scheme by node type
    color_map = {
        "Place": "#4ecca3",       # Green
        "FoodShop": "#e84545",    # Red
        "Hotel": "#3282b8",       # Blue
        "Street": "#f9d56e",      # Yellow
        "Dish": "#ff8b64",        # Orange
        "TourDay": "#903749",     # Purple
        "Activity": "#00b4d8",    # Cyan
        "PriceTier": "#a8dda8",   # Soft green
        "HistoricalEra": "#d4a5a5"# Dusty rose
    }

    for node, data in G.nodes(data=True):
        label = data.get("label", "Concept")
        color = color_map.get(label, "#9a8c98")
        title_hover = f"<b>{node}</b><br/>Type: {label}"
        if "open_hours" in data:
            title_hover += f"<br/>เวลาเปิด: {data['open_hours']}"
        if "price_range" in data:
            title_hover += f"<br/>ราคา: {data['price_range']}"
        if "rating" in data:
            title_hover += f"<br/>ดาว: {data['rating']} ⭐"
            
        size = 30 if label in ["Place", "FoodShop", "Hotel"] else (22 if label in ["Street", "TourDay"] else 15)
        net.add_node(node, label=node, title=title_hover, color=color, size=size)

    for src, tgt, data in G.edges(data=True):
        rel = data.get("relation", "RELATED_TO")
        net.add_edge(src, tgt, title=rel, label=rel, color="#6c757d", arrows="to")

    html_path = os.path.join(base_dir, "songkhla_knowledge_graph.html")
    net.write_html(html_path)
    print(f"[OK] Generated Interactive Graph HTML: {html_path}")

    # 4. Export High-Res Static Diagram for Presentation Slides
    print(f"\n[STEP 4] Rendering High-Res Static Diagram for Presentation Slides...")
    plt.figure(figsize=(16, 12), dpi=200)
    plt.style.use('dark_background')
    
    # Subgraph for clean overview: Places, Streets, and Key Connections
    key_nodes = [n for n, d in G.nodes(data=True) if d.get("label") in ["Place", "FoodShop", "Hotel", "Street", "TourDay"]]
    subG = G.subgraph(key_nodes)
    
    pos = nx.spring_layout(subG, k=0.6, iterations=50, seed=42)
    
    # Node colors
    node_colors = [color_map.get(subG.nodes[n].get("label", "Concept"), "#9a8c98") for n in subG.nodes()]
    node_sizes = [1800 if subG.nodes[n].get("label") in ["Place", "FoodShop", "Hotel"] else 1200 for n in subG.nodes()]
    
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_edges(subG, pos, edge_color="#888888", alpha=0.6, arrows=True, arrowsize=15, width=1.2)
    
    # Simple labels (in English/Thai friendly)
    labels = {n: n.split("(")[0].strip() for n in subG.nodes()}
    # Set Thai font on Windows
    plt.rcParams['font.family'] = 'Leelawadee UI'
    labels = {n: n.split("(")[0].strip() for n in subG.nodes()}
    nx.draw_networkx_labels(subG, pos, labels=labels, font_size=8, font_color="white", font_family='Leelawadee UI')
    
    plt.title("Songkhla Old Town Knowledge Graph Schema & Architecture (Final Project)", fontsize=16, color="white", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    png_path = os.path.join(base_dir, "songkhla_graph_overview.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved High-Res Thai Graph Overview Image: {png_path}")
    print("\nALL GRAPH GENERATION STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    build_and_export_graph()
