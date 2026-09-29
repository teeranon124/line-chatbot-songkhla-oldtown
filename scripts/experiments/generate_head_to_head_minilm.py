import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Set font family
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Garuda', 'Leelawadee UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#F8FAFC')

# Title & Subtitle
plt.suptitle('การเปรียบเทียบเชิงวิศวกรรมแบบหมัดต่อหมัด (Head-to-Head Shootout)\nMiniLM-L12 (Multilingual) vs WangchanBERTa (Thai Native)', 
             fontsize=19, fontweight='bold', color='#0F172A', y=0.96)
plt.figtext(0.5, 0.89, 'ทำไมเราถึงเลือก WangchanBERTa ทั้งที่ MiniLM-L12 เป็นโมเดลยอดนิยมที่เร็วและเบา?', 
           ha='center', fontsize=12, color='#475569')

# Define comparison metrics
metrics = [
    {
        "name": "ขนาดโมเดล (Model Size)",
        "unit": "Parameters (ล้านตัว)",
        "minilm_val": "117.7 M",
        "wangchan_val": "105.8 M",
        "winner": "wangchan",
        "badge_wangchan": "เบากว่า 10%",
        "badge_minilm": "",
        "desc": "WangchanBERTa เบากว่าเล็กน้อย ประหยัด RAM ฝั่งเซิร์ฟเวอร์"
    },
    {
        "name": "มิติเวกเตอร์ (Embedding Dimension)",
        "unit": "Dimensions (d)",
        "minilm_val": "384 d",
        "wangchan_val": "768 d",
        "winner": "wangchan",
        "badge_wangchan": "ละเอียดกว่า 2x",
        "badge_minilm": "",
        "desc": "768-dim เก็บความหมายเชิงลึกและบริบทประวัติศาสตร์สงขลาได้ดีกว่า 384-dim"
    },
    {
        "name": "การแตกคำภาษาไทย (Token Fragmentation)",
        "unit": "Tokens ต่อ 1 คำเฉพาะ",
        "minilm_val": "5.2 tokens",
        "wangchan_val": "4.4 tokens",
        "winner": "wangchan",
        "badge_wangchan": "ดีกว่า (แตกคำน้อยกว่า)",
        "badge_minilm": "แตกคำเละ (8 tokens)",
        "desc": "MiniLM แตกคำเฉพาะเละ เช่น 'แต้เฮี้ยงอิ้ว' แตก 8 ท่อน ขณะที่ Wangchan เกาะกลุ่มดีกว่า"
    },
    {
        "name": "ความเร็วในการ Encode บน CPU",
        "unit": "Milliseconds (Latency)",
        "minilm_val": "76.4 ms",
        "wangchan_val": "164.8 ms",
        "winner": "minilm",
        "badge_wangchan": "Real-time (< 0.2s)",
        "badge_minilm": "เร็วกว่า 2.1x",
        "desc": "MiniLM เร็วกว่าเพราะเวกเตอร์ 384d แต่ 164.8 ms ของ Wangchan ก็อยู่ในเกณฑ์ Real-time สบายๆ"
    },
    {
        "name": "ความแม่นยำเวกเตอร์เดี่ยว (Dense Standalone MRR)",
        "unit": "MRR Score (0.0 - 1.0)",
        "minilm_val": "0.7061",
        "wangchan_val": "0.8944",
        "winner": "wangchan",
        "badge_wangchan": "ชนะขาด (+0.188)",
        "badge_minilm": "Hit@1 ตกเหลือ 60%",
        "desc": "เวกเตอร์เพียวๆ MiniLM หลุดคำเฉพาะไปเยอะมาก (Hit@1 ได้แค่ 60% vs Wangchan 86.7%)"
    },
    {
        "name": "ผลลัพธ์เมื่อทำ Hybrid (+BM25)",
        "unit": "Hit@5 Recall Rate",
        "minilm_val": "93.3%",
        "wangchan_val": "93.3%",
        "winner": "tie",
        "badge_wangchan": "พุ่ง +13.3% เท่ากัน",
        "badge_minilm": "พุ่งเท่ากัน",
        "desc": "เมื่อมี BM25 มาช่วย ทั้งคู่ทำ Hit@5 ได้ 93.3% เท่ากัน แต่ Wangchan ชนะขาดเรื่องโครงสร้างภาษาไทย"
    }
]

# Draw cards
y_starts = np.linspace(0.81, 0.14, len(metrics))
box_height = 0.088

for i, m in enumerate(metrics):
    y = y_starts[i]
    
    # Outer card
    rect = patches.FancyBboxPatch((0.04, y - box_height + 0.015), 0.92, box_height,
                                  boxstyle="round,pad=0.01,rounding_size=0.015",
                                  linewidth=1, edgecolor='#E2E8F0', facecolor='#FFFFFF')
    ax.add_patch(rect)
    
    # Metric Name
    ax.text(0.06, y - 0.018, m["name"], fontsize=11.5, fontweight='bold', color='#1E293B', va='top')
    ax.text(0.06, y - 0.048, m["desc"], fontsize=9.2, color='#64748B', va='top')
    
    # MiniLM Box (Column 1)
    minilm_bg = '#F1F5F9' if m["winner"] != "minilm" else '#DCFCE7'
    minilm_border = '#CBD5E1' if m["winner"] != "minilm" else '#22C55E'
    rect_m = patches.FancyBboxPatch((0.48, y - box_height + 0.022), 0.20, box_height - 0.015,
                                    boxstyle="round,pad=0.008,rounding_size=0.01",
                                    linewidth=1.2, edgecolor=minilm_border, facecolor=minilm_bg)
    ax.add_patch(rect_m)
    ax.text(0.58, y - 0.028, m["minilm_val"], fontsize=11.5, fontweight='bold', 
            color='#0F172A', ha='center', va='center')
    if m["badge_minilm"]:
        badge_col = '#16A34A' if m["winner"] == "minilm" else '#DC2626'
        ax.text(0.58, y - 0.054, m["badge_minilm"], fontsize=8.5, fontweight='bold',
                color=badge_col, ha='center', va='center')

    # Wangchan Box (Column 2)
    wang_bg = '#EFF6FF' if m["winner"] != "wangchan" else '#DCFCE7'
    wang_border = '#93C5FD' if m["winner"] != "wangchan" else '#22C55E'
    rect_w = patches.FancyBboxPatch((0.72, y - box_height + 0.022), 0.22, box_height - 0.015,
                                    boxstyle="round,pad=0.008,rounding_size=0.01",
                                    linewidth=1.5, edgecolor=wang_border, facecolor=wang_bg)
    ax.add_patch(rect_w)
    ax.text(0.83, y - 0.028, m["wangchan_val"], fontsize=11.5, fontweight='bold', 
            color='#1E40AF' if m["winner"] != "wangchan" else '#15803D', ha='center', va='center')
    if m["badge_wangchan"]:
        ax.text(0.83, y - 0.054, m["badge_wangchan"], fontsize=8.5, fontweight='bold',
                color='#15803D', ha='center', va='center')

# Column Headers
ax.text(0.58, 0.855, 'MiniLM-L12 (Multilingual)', fontsize=12, fontweight='bold', color='#475569', ha='center')
ax.text(0.83, 0.855, 'WangchanBERTa (โมเดลที่เราเลือก)', fontsize=12, fontweight='bold', color='#1D4ED8', ha='center')

# Bottom Summary banner
summary_rect = patches.FancyBboxPatch((0.04, 0.008), 0.92, 0.042,
                                      boxstyle="round,pad=0.008,rounding_size=0.01",
                                      linewidth=1.2, edgecolor='#3B82F6', facecolor='#EFF6FF')
ax.add_patch(summary_rect)
ax.text(0.5, 0.029, 
        'สรุปวิศวกรรม: MiniLM-L12 เร็วกว่าแต่พลาดเรื่องคำเฉพาะภาษาไทย (MRR ต่ำกว่าถึง 0.188) | WangchanBERTa แม่นยำกว่า แตกคำน้อยกว่า และเมื่อเสริม BM25 ได้ประสิทธิภาพสูงสุด',
        fontsize=10.0, fontweight='bold', color='#1E40AF', ha='center', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
output_path = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart1_wangchan_vs_minilm_head_to_head.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {output_path}")
