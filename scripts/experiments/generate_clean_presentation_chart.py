import matplotlib.pyplot as plt
import numpy as np

# Set font for academic presentation
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 16:9 presentation aspect ratio
fig, ax = plt.subplots(figsize=(14, 7.5), facecolor='#FFFFFF', dpi=300)
ax.set_facecolor('#FFFFFF')

# Models: Systematic selection across Dimensions & Language Families
models = [
    'BGE-M3\n(Multilingual 1024d)',
    'WangchanBERTa-ConGen\n(Thai 768d)',
    'MiniLM-L12\n(Multilingual 384d)',
    'WangchanBERTa-SimCSE\n(Thai 768d)',
    'PhayaThaiBERT-SCT\n(Thai 768d)'
]

# Hit@5 Recall Rate (%)
dense_scores = [72.2, 66.7, 77.8, 61.1, 61.1]
hybrid_scores = [83.3, 83.3, 77.8, 77.8, 66.7]
diffs = [11.1, 16.7, 0.0, 16.7, 5.6]

x = np.arange(len(models))
width = 0.32

# Clean presentation bars
bar1 = ax.bar(x - width/2, dense_scores, width, label='แบบที่ 1: Dense Only (ใช้เวกเตอร์โมเดลเพียวๆ ไม่ผสม BM25)', 
              color='#5B6777', edgecolor='#334155', linewidth=1.0, zorder=3)
bar2 = ax.bar(x + width/2, hybrid_scores, width, label='แบบที่ 2: Hybrid Retrieval (นำโมเดลมารวมกับ BM25 + RRF เหมือนกันทุกตัว)', 
              color='#2563EB', edgecolor='#1D4ED8', linewidth=1.0, zorder=3)

# Axis styling
ax.set_ylabel('Hit@5 Recall Rate (%) [ยิ่งสูงยิ่งดึงคำตอบได้ครบ]', fontsize=12, weight='bold', color='#1E293B', labelpad=12)
ax.set_title('ผลการทดสอบเปรียบเทียบโมเดลอย่างยุติธรรม (ทุกโมเดลทดสอบทั้ง Dense เดี่ยวๆ และทำ Hybrid เหมือนกันหมด)\n', 
             fontsize=15, weight='bold', color='#0F172A', pad=15)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11, weight='bold', color='#1E293B')
ax.set_ylim(0, 112)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='y', labelsize=10, colors='#475569')

# Clean subtle grid lines
ax.grid(axis='y', linestyle='--', alpha=0.6, color='#E2E8F0', zorder=0)
ax.set_axisbelow(True)

# Spines styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Value labels and Badges
for i in range(len(models)):
    d_val = dense_scores[i]
    h_val = hybrid_scores[i]
    diff = diffs[i]
    
    # Values above bars
    ax.text(x[i] - width/2, d_val + 1.5, f'{d_val:.1f}%', ha='center', va='bottom', 
            fontsize=10.5, weight='bold', color='#334155', zorder=4)
    ax.text(x[i] + width/2, h_val + 1.5, f'{h_val:.1f}%', ha='center', va='bottom', 
            fontsize=10.5, weight='bold', color='#1D4ED8', zorder=4)
    
    # Badge above pair
    max_val = max(d_val, h_val)
    badge_y = max_val + 7.5
    
    if diff > 0:
        badge_text = f'ผลต่าง: +{diff:.1f}%'
        box_edge = '#22C55E'
        box_face = '#F0FDF4'
        text_color = '#15803D'
    else:
        badge_text = f'ผลต่าง: 0.0% (เท่าเดิม)'
        box_edge = '#94A3B8'
        box_face = '#F8FAFC'
        text_color = '#64748B'
        
    ax.text(x[i], badge_y, badge_text, ha='center', va='center', fontsize=9.5, weight='bold', color=text_color,
            bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.3', facecolor=box_face, edgecolor=box_edge, linewidth=1.2), zorder=4)

# Legend
legend = ax.legend(loc='upper right', frameon=True, fontsize=10.5)
legend.get_frame().set_facecolor('#F8FAFC')
legend.get_frame().set_edgecolor('#CBD5E1')
legend.get_frame().set_boxstyle('round,pad=0.5,rounding_size=0.2')

plt.tight_layout()
out_file = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart_model_comparison_presentation.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f'Successfully generated clean presentation chart at: {out_file}')
