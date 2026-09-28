# -*- coding: utf-8 -*-
import pickle
from collections import Counter

with open("data/songkhla_graph.pkl", "rb") as f:
    G = pickle.load(f)

print("=" * 60)
print("📊 FINAL AUDIT OF KNOWLEDGE GRAPH DATABASE")
print("=" * 60)
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")

print("\n🏷️ NODE LABELS DISTRIBUTION (Ontology Classes):")
for label, cnt in Counter([d.get("label") for n, d in G.nodes(data=True)]).most_common():
    print(f"  - {label:15s}: {cnt:2d} nodes")

print("\n🔗 EDGE RELATIONS DISTRIBUTION (Allowed Relations):")
for rel, cnt in Counter([d.get("relation") for u, v, d in G.edges(data=True)]).most_common():
    print(f"  - {rel:15s}: {cnt:2d} edges")

isolated = [n for n in G.nodes() if G.degree(n) == 0]
print(f"\n🏝️ ISOLATED NODES (Degree = 0): {len(isolated)}")
if isolated:
    print("  Isolated items:", isolated)
else:
    print("  ✅ ZERO isolated nodes! 100% of nodes are meaningfully connected in the network!")

print("\n🏛️ HUB NODES (Top 5 Highest Degree Nodes):")
for n, deg in sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]:
    lbl = G.nodes[n].get("label", "Concept")
    print(f"  - [{lbl}] {n}: {deg} connections (in={G.in_degree(n)}, out={G.out_degree(n)})")

print("\n💎 VERIFYING RICH ATTRIBUTES ON CONNECTED PLACE NODES:")
sample_places = ["ร้านเกียดฟั่ง", "ร้านไอติมโอ่ง", "โรงสีแดง หับโห้หิ้น", "โรงแรมสงขลาแต่แรก"]
for name in sample_places:
    d = G.nodes.get(name, {})
    out_edges = [f"{v} ({ed.get('relation')})" for _, v, ed in G.out_edges(name, data=True)]
    print(f"* {name} [{d.get('label')}] (Degree: {G.degree(name)}):")
    print(f"    Street: {d.get('street')} | Open: {d.get('open_hours')} | Rating: {d.get('rating')} ⭐")
    print(f"    Connected Out: {out_edges[:3]}")
