import matplotlib.pyplot as plt
import numpy as np

# Set font for academic presentation
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 16:9 Widescreen aspect ratio (Two side-by-side panels)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), facecolor='#FFFFFF', dpi=300)

models = [
    'WangchanBERTa\n(768d | 426MB)',
    'E5-Base\n(768d | 1.1GB)',
    'MiniLM-L12\n(384d | 480MB)',
    'E5-Small\n(384d | 471MB)'
]

x = np.arange(len(models))
width = 0.34

# Data: Before vs After BM25
# 1. Hit@1 Data
dense_hit1 = [75.0, 85.0, 60.0, 75.0]
hybrid_hit1 = [85.0, 80.0, 70.0, 70.0]
diff_hit1 = [h - d for d, h in zip(dense_hit1, hybrid_hit1)]

# 2. Hit@5 Data
dense_hit5 = [95.0, 95.0, 85.0, 100.0]
hybrid_hit5 = [95.0, 100.0, 95.0, 100.0]
diff_hit5 = [h - d for d, h in zip(dense_hit5, hybrid_hit5)]

def style_subplot(ax, title, dense_data, hybrid_data, diff_data, metric_name):
    ax.set_facecolor('#FFFFFF')
    
    # Bars
    b1 = ax.bar(x - width/2, dense_data, width, label='ก่อนใช้: Dense Only (เวกเตอร์เดี่ยว)', 
                color='#64748B', edgecolor='#334155', linewidth=1.0, zorder=3)
    b2 = ax.bar(x + width/2, hybrid_data, width, label='หลังใช้: Hybrid (+BM25 RRF)', 
                color='#2563EB', edgecolor='#1D4ED8', linewidth=1.0, zorder=3)
    
    ax.set_title(title, fontsize=13, weight='bold', color='#0F172A', pad=15)
    ax.set_ylabel(f'{metric_name} (%)', fontsize=11, weight='bold', color='#1E293B', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, weight='bold', color='#1E293B')
    ax.set_ylim(0, 126)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis='y', labelsize=10, colors='#475569')
    
    # Clean subtle grid lines
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#E2E8F0', zorder=0)
    ax.set_axisbelow(True)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Labels and badges
    for i in range(len(models)):
        d_val = dense_data[i]
        h_val = hybrid_data[i]
        diff = diff_data[i]
        
        # Values above bars
        ax.text(x[i] - width/2, d_val + 1.5, f'{d_val:.1f}%', ha='center', va='bottom', 
                fontsize=9.5, weight='bold', color='#334155', zorder=4)
        ax.text(x[i] + width/2, h_val + 1.5, f'{h_val:.1f}%', ha='center', va='bottom', 
                fontsize=9.5, weight='bold', color='#1D4ED8', zorder=4)
        
        # Badge
        max_val = max(d_val, h_val)
        badge_y = max_val + 8.0
        
        if diff > 0:
            badge_text = f'+{diff:.1f}%'
            box_edge = '#16A34A'
            box_face = '#F0FDF4'
            text_color = '#15803D'
        elif diff == 0:
            badge_text = '0.0%'
            box_edge = '#94A3B8'
            box_face = '#F8FAFC'
            text_color = '#64748B'
        else:
            badge_text = f'{diff:.1f}%'
            box_edge = '#DC2626'
            box_face = '#FEF2F2'
            text_color = '#B91C1C'
            
        ax.text(x[i], badge_y, badge_text, ha='center', va='center', fontsize=9.5, weight='bold', color=text_color,
                bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.3', facecolor=box_face, edgecolor=box_edge, linewidth=1.2), zorder=4)

# Render Left (Hit@1) and Right (Hit@5)
style_subplot(ax1, '(ก) ความแม่นยำอันดับที่ 1 (Hit@1 Recall Rate)\n[ชี้วัดว่าคำตอบขึ้นมาเป็นอันดับแรกให้ LLM อ่านหรือไม่]', 
              dense_hit1, hybrid_hit1, diff_hit1, 'Hit@1 Recall Rate')
style_subplot(ax2, '(ข) ความครอบคลุมใน 5 อันดับแรก (Hit@5 Recall Rate)\n[ชี้วัดว่าคำตอบติดเข้ามาใน Context รวมหรือไม่]', 
              dense_hit5, hybrid_hit5, diff_hit5, 'Hit@5 Recall Rate')

# Global Legend
handles, labels = ax1.get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, 
                    frameon=True, fontsize=11)
legend.get_frame().set_facecolor('#F8FAFC')
legend.get_frame().set_edgecolor('#CBD5E1')
legend.get_frame().set_boxstyle('round,pad=0.4,rounding_size=0.2')

plt.suptitle('การเปรียบเทียบก่อนและหลังใช้ BM25 (Dense Only vs Hybrid Retrieval) บนคลังข้อมูลสงขลา\n', 
             fontsize=15, weight='bold', color='#0F172A', y=1.03)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_file = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart_dense_vs_hybrid_hit1_and_hit5.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
plt.close()
print(f'Successfully generated dual-panel comparison chart at: {out_file}')
