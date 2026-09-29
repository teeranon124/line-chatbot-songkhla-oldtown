import matplotlib.pyplot as plt
import numpy as np

# Set font for academic presentation
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 16:9 presentation aspect ratio
fig, ax1 = plt.subplots(figsize=(15, 8.5), facecolor='#FFFFFF', dpi=300)
ax1.set_facecolor('#FFFFFF')

# Models and Data from Empirical Songkhla Old Town Benchmark
models = [
    'WangchanBERTa-ConGen\n(768d | RAM ~426MB)',
    'Multilingual-E5-Base\n(768d | RAM ~1.1GB)',
    'MiniLM-L12\n(384d | RAM ~480MB)',
    'Multilingual-E5-Small\n(384d | RAM ~471MB)'
]

hit1_scores = [85.0, 80.0, 70.0, 70.0]
hit5_scores = [95.0, 100.0, 95.0, 100.0]
mrr_scores = [0.8883, 0.8917, 0.8097, 0.8083]

x = np.arange(len(models))
width = 0.32

# Bars for Hit@1 and Hit@5
bar1 = ax1.bar(x - width/2, hit1_scores, width, label='Hit@1 (คำตอบอยู่อันดับ 1 เป๊ะ - ส่งผลต่อ LLM สูงสุด)', 
               color='#2563EB', edgecolor='#1D4ED8', linewidth=1.0, zorder=3)
bar2 = ax1.bar(x + width/2, hit5_scores, width, label='Hit@5 (คำตอบติด 1 ใน 5 อันดับแรก)', 
               color='#64748B', edgecolor='#475569', linewidth=1.0, zorder=3)

# Axis 1 Styling (Percentage)
ax1.set_ylabel('ความแม่นยำ Recall Rate (%)', fontsize=12, weight='bold', color='#1E293B', labelpad=12)
ax1.set_title('การเปรียบเทียบประสิทธิภาพเชิงลึกระหว่าง Embedding Models ในระบบ Hybrid RAG (+BM25)\n(วัดผลบนคลังข้อมูลเมืองเก่าสงขลา 21 Chunks ด้วยตัวชี้วัด Hit@1, Hit@5 และ MRR)', 
              fontsize=14, weight='bold', color='#0F172A', pad=18)

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=11, weight='bold', color='#1E293B')
ax1.set_ylim(0, 122)
ax1.set_yticks(np.arange(0, 101, 20))
ax1.tick_params(axis='y', labelsize=10, colors='#475569')

# Clean subtle grid lines
ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#E2E8F0', zorder=0)
ax1.set_axisbelow(True)

# Spines styling
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#CBD5E1')
ax1.spines['bottom'].set_color('#CBD5E1')
ax1.spines['left'].set_linewidth(1.2)
ax1.spines['bottom'].set_linewidth(1.2)

# Value labels on top of bars
for i in range(len(models)):
    # Hit@1 label
    ax1.text(x[i] - width/2, hit1_scores[i] + 1.5, f'{hit1_scores[i]:.1f}%', ha='center', va='bottom', 
             fontsize=10.5, weight='bold', color='#1E40AF', zorder=4)
    # Hit@5 label
    ax1.text(x[i] + width/2, hit5_scores[i] + 1.5, f'{hit5_scores[i]:.1f}%', ha='center', va='bottom', 
             fontsize=10.5, weight='bold', color='#334155', zorder=4)
    
    # MRR Badge above pair
    max_val = max(hit1_scores[i], hit5_scores[i])
    badge_y = max_val + 8.5
    mrr_val = mrr_scores[i]
    
    # Highlight highest Hit@1 & top MRR
    if i == 0:
        badge_text = f'MRR: {mrr_val:.4f} (Hit@1 สูงสุด)'
        box_edge = '#2563EB'
        box_face = '#EFF6FF'
        text_color = '#1D4ED8'
    else:
        badge_text = f'MRR: {mrr_val:.4f}'
        box_edge = '#059669'
        box_face = '#ECFDF5'
        text_color = '#047857'
        
    ax1.text(x[i], badge_y, badge_text, ha='center', va='center', fontsize=9.5, weight='bold', color=text_color,
             bbox=dict(boxstyle='round,pad=0.4,rounding_size=0.3', facecolor=box_face, edgecolor=box_edge, linewidth=1.2), zorder=4)

# Legend
legend = ax1.legend(loc='upper right', frameon=True, fontsize=10.5)
legend.get_frame().set_facecolor('#F8FAFC')
legend.get_frame().set_edgecolor('#CBD5E1')
legend.get_frame().set_boxstyle('round,pad=0.5,rounding_size=0.2')

plt.tight_layout()
out_file = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart_embedding_deep_comparison.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f'Successfully generated deep comparison chart at: {out_file}')
