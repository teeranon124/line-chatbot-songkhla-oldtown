# -*- coding: utf-8 -*-
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password1234"))

with driver.session() as session:
    print("=" * 60)
    print("🟢 NEO4J LIVE DATABASE INSPECTION")
    print("=" * 60)
    
    total_nodes = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
    total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]
    print(f"Total Nodes: {total_nodes}")
    print(f"Total Relationships: {total_rels}\n")
    
    print("🏷️ NODE LABELS:")
    res = session.run("MATCH (n) RETURN coalesce(labels(n)[0], 'Concept') AS label, count(n) AS cnt ORDER BY cnt DESC")
    for r in res:
        print(f"  - {r['label']:15s}: {r['cnt']:2d} nodes")
        
    print("\n🔗 RELATIONSHIP TYPES:")
    res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
    for r in res:
        print(f"  - {r['rel']:15s}: {r['cnt']:2d} relations")
        
    print("\n🍜 SAMPLE CYPHER: FOOD SHOPS & DISHES:")
    res = session.run("MATCH (s)-[:SERVES]->(d) RETURN s.name AS shop, collect(d.name) AS dishes")
    for r in res:
        dishes = ", ".join(r["dishes"])
        print(f"  * {r['shop']}: {dishes}")
        
    print("\n📍 SAMPLE CYPHER: STREET LOCATIONS:")
    res = session.run("MATCH (p)-[:LOCATED_ON]->(st) RETURN st.name AS street, count(p) AS places_count, collect(p.name)[..4] AS sample_places")
    for r in res:
        samples = ", ".join(r["sample_places"])
        print(f"  * {r['street']} ({r['places_count']} แห่ง): {samples}")

driver.close()
