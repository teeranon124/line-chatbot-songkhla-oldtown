import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean academic styling
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='#FFFFFF')
ax.set_facecolor('#FFFFFF')
ax.axis('off')

# Title
fig.text(0.5, 0.95, 'สถาปัตยกรรมกระบวนการค้นหาและรีแรงค์กิ้ง (Retrieval & Re-ranking Pipeline)', 
         ha='center', fontsize=18, weight='bold', color='#1E3A8A')
fig.text(0.5, 0.90, 'การแก้ปัญหา Score Scale Mismatch ด้วยสูตร Reciprocal Rank Fusion (RRF k=60) ผสาน Graph Enrichment', 
         ha='center', fontsize=12, color='#475569')

# Draw Pipeline Flow Boxes
# Box 1: User Query
q_box = patches.FancyBboxPatch((0.05, 0.40), 0.14, 0.16, boxstyle='round,pad=0.02,rounding_size=0.02', 
                              facecolor='#EFF6FF', edgecolor='#3B82F6', linewidth=1.5)
ax.add_patch(q_box)
ax.text(0.12, 0.50, 'คำถามผู้ใช้ (Query)\n"ร้านแต้เฮี้ยงอิ๋ว\nเปิดกี่โมง"', ha='center', va='center', fontsize=10.5, weight='bold', color='#1E3A8A')

# Arrows to dual tracks
ax.annotate('', xy=(0.26, 0.65), xytext=(0.19, 0.52), arrowprops=dict(arrowstyle="->", color='#3B82F6', lw=2))
ax.annotate('', xy=(0.26, 0.31), xytext=(0.19, 0.44), arrowprops=dict(arrowstyle="->", color='#3B82F6', lw=2))

# Track 1: Dense Retrieval
t1_box = patches.FancyBboxPatch((0.26, 0.56), 0.20, 0.18, boxstyle='round,pad=0.02,rounding_size=0.02', 
                               facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.2)
ax.add_patch(t1_box)
ax.text(0.36, 0.68, '1. Dense Retrieval (Semantic)', ha='center', va='center', fontsize=10.5, weight='bold', color='#0F172A')
ax.text(0.36, 0.61, '• WangchanBERTa (768d)\n• FAISS Index Search\n• Score: Cosine (-1 ถึง +1)', ha='center', va='center', fontsize=9, color='#475569')

# Track 2: Sparse Retrieval
t2_box = patches.FancyBboxPatch((0.26, 0.22), 0.20, 0.18, boxstyle='round,pad=0.02,rounding_size=0.02', 
                               facecolor='#F8FAFC', edgecolor='#64748B', linewidth=1.2)
ax.add_patch(t2_box)
ax.text(0.36, 0.34, '2. Sparse Retrieval (Lexical)', ha='center', va='center', fontsize=10.5, weight='bold', color='#0F172A')
ax.text(0.36, 0.27, '• PyThaiNLP newmm\n• BM25Okapi Index\n• Score: TF-IDF (0 ถึง 20+)', ha='center', va='center', fontsize=9, color='#475569')

# Problem Callout (Score Mismatch)
warn_box = patches.FancyBboxPatch((0.26, 0.05), 0.20, 0.11, boxstyle='round,pad=0.015,rounding_size=0.01', 
                                 facecolor='#FEF2F2', edgecolor='#EF4444', linewidth=1.2)
ax.add_patch(warn_box)
ax.text(0.36, 0.105, '[!] ปัญหา Score Scale Mismatch\nDense เป็น Cosine (-1 ถึง 1)\nBM25 เป็น TF-IDF (0 ถึง 20+)\nถ้าบวกตรงๆ BM25 จะกลืน Dense หมด!', 
        ha='center', va='center', fontsize=8.5, weight='bold', color='#B91C1C')

# Arrows to RRF
ax.annotate('', xy=(0.53, 0.52), xytext=(0.46, 0.65), arrowprops=dict(arrowstyle="->", color='#1D4ED8', lw=2))
ax.annotate('', xy=(0.53, 0.44), xytext=(0.46, 0.31), arrowprops=dict(arrowstyle="->", color='#1D4ED8', lw=2))

# Center Box: Reciprocal Rank Fusion (RRF Engine)
rrf_box = patches.FancyBboxPatch((0.53, 0.28), 0.22, 0.40, boxstyle='round,pad=0.02,rounding_size=0.02', 
                                facecolor='#EFF6FF', edgecolor='#1D4ED8', linewidth=2.0)
ax.add_patch(rrf_box)
ax.text(0.64, 0.62, 'Reciprocal Rank Fusion (RRF)\nกลไกการรีแรงค์กิ้ง (k=60)', ha='center', va='center', fontsize=12, weight='bold', color='#1E3A8A')
ax.text(0.64, 0.48, r'$RRF(d) = \sum_{m} \frac{w_m}{60 + r_m(d)}$', ha='center', va='center', fontsize=12, weight='bold', color='#1D4ED8')
ax.text(0.64, 0.36, '• ใช้ "อันดับ (Rank)" แทนคะแนนดิบ\n• k=60 ลดอิทธิพลของ Outlier\n• น้ำหนัก Dense 0.50 + Sparse 0.50\n• สลับจัดอันดับใหม่ (Unified Rank)', ha='center', va='center', fontsize=9, color='#1E293B')

# Arrow to Graph Enrichment
ax.annotate('', xy=(0.80, 0.48), xytext=(0.75, 0.48), arrowprops=dict(arrowstyle="->", color='#059669', lw=2.5))

# Final Box: Graph Enrichment & Output
final_box = patches.FancyBboxPatch((0.80, 0.28), 0.16, 0.40, boxstyle='round,pad=0.02,rounding_size=0.02', 
                                  facecolor='#F0FDF4', edgecolor='#059669', linewidth=2.0)
ax.add_patch(final_box)
ax.text(0.88, 0.62, 'Knowledge Graph\nEnrichment', ha='center', va='center', fontsize=12, weight='bold', color='#166534')
ax.text(0.88, 0.47, '• ดึง Triples ที่เกี่ยวข้อง\n• เติมข้อมูลถนน & ความสัมพันธ์\n• Dynamic Top-K Selector\n• ส่งบริบทสมบูรณ์ให้ LLM', ha='center', va='center', fontsize=9.2, color='#15803D')
ax.text(0.88, 0.33, 'ผลลัพธ์:\nHit@3 = 100%\nMRR = 0.9250', ha='center', va='center', fontsize=10.5, weight='bold', color='#047857')

plt.tight_layout()
out_path = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart4_reranking_architecture.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f'Generated Re-ranking Architecture to {out_path}')
