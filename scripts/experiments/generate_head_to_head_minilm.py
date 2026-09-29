import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Set font family
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Garuda', 'Leelawadee UI', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300)
ax.set_facecolor('#F8FAFC')
fig.patch.set_facecolor('#F8FAFC')

# Title & Subtitle
plt.suptitle('การเปรียบเทียบเชิงลึก: MiniLM-L12 vs WangchanBERTa\nไขข้อข้องใจ: ในกราฟ Hit@5 MiniLM ได้ 93.3% สูงกว่า Wangchan (80%) แล้วทำไมเราถึงเลือก WangchanBERTa?', 
             fontsize=17, fontweight='bold', color='#0F172A', y=0.96)
plt.figtext(0.5, 0.90, 'วิเคราะห์ความแตกต่างระหว่าง "Hit@5 (ขอแค่ติด 1 ใน 5)" กับ "Hit@1 และ MRR (คำตอบต้องอยู่อันดับ 1)"', 
           ha='center', fontsize=12, color='#475569')

# Metrics matching exp1_bert_vs_bert_bm25_results.json
metrics = [
    {
        "name": "1. ความแม่นยำอันดับแรก (Hit@1 Accuracy)",
        "minilm_val": "40.0% (6/15 ข้อ)",
        "wangchan_val": "53.3% (8/15 ข้อ)",
        "winner": "wangchan",
        "badge_wangchan": "ชนะ (+13.3%)",
        "badge_minilm": "ตกไปอยู่อันดับล่างๆ",
        "desc": "Wangchan ดึงคำตอบที่ถูกต้องขึ้นเป็น 'อันดับ 1' ได้แม่นยำกว่า MiniLM อย่างชัดเจน"
    },
    {
        "name": "2. คะแนนเฉลี่ยตามอันดับ (Dense Standalone MRR)",
        "minilm_val": "0.6002",
        "wangchan_val": "0.6730",
        "winner": "wangchan",
        "badge_wangchan": "ชนะ (+0.0728)",
        "badge_minilm": "คะแนนต่ำกว่า",
        "desc": "MRR วัดว่าคำตอบอยู่หัวตารางแค่ไหน Wangchan ได้คะแนนเฉลี่ยสูงกว่าเพราะคำตอบไม่หล่นไปไกล"
    },
    {
        "name": "3. การติด 1 ใน 5 อันดับแรก (Dense Hit@5) [กราฟฟุตซอล]",
        "minilm_val": "93.3% (14/15 ข้อ)",
        "wangchan_val": "80.0% (12/15 ข้อ)",
        "winner": "minilm",
        "badge_wangchan": "ติดอันดับล่าง (ตามหลัง)",
        "badge_minilm": "ชนะใน Hit@5",
        "desc": "MiniLM กวาดติด 5 อันดับแรกได้มากกว่า แต่คำตอบมักไปกองอยู่ที่อันดับ 3, 4, 5 (ไม่ใช่หัวตาราง)"
    },
    {
        "name": "4. เมื่อทำ Hybrid รวมกับ BM25 (+BM25 Hit@5)",
        "minilm_val": "93.3% (เพิ่ม +0.0%)",
        "wangchan_val": "93.3% (พุ่ง +13.3%)",
        "winner": "tie",
        "badge_wangchan": "ก้าวกระโดดขึ้นมาเท่ากัน!",
        "badge_minilm": "ไม่พัฒนาขึ้น (+0%)",
        "desc": "พอมี BM25 ช่วยดึงคำเฉพาะ Wangchan พุ่งจาก 80% เป็น 93.3% เท่า MiniLM ทันที"
    },
    {
        "name": "5. การแตกคำภาษาไทย (Token Fragmentation)",
        "minilm_val": "5.2 tokens / คำ",
        "wangchan_val": "4.4 tokens / คำ",
        "winner": "wangchan",
        "badge_wangchan": "รักษาคำไทยดีกว่า",
        "badge_minilm": "แตกคำเละ (แต้เฮี้ยงอิ้ว 8 ท่อน)",
        "desc": "MiniLM แตกคำเฉพาะสงขลาละเอียดเกินไปจนเสียความหมาย ขณะที่ Wangchan เกาะกลุ่มคำไทยได้ดี"
    },
    {
        "name": "6. ขนาดโมเดล และ ความเร็ว CPU",
        "minilm_val": "117.7M (เร็ว 76.4 ms)",
        "wangchan_val": "105.8M (164.8 ms)",
        "winner": "tie",
        "badge_wangchan": "เบากว่า 10% (105M)",
        "badge_minilm": "Encode เร็วกว่า",
        "desc": "MiniLM เร็วกว่า แต่ Wangchan เล็กกว่า (105.8M) และ 164 ms ก็เร็วกว่าเกณฑ์ Real-time (<0.5s)"
    }
]

# Draw cards
y_starts = np.linspace(0.82, 0.14, len(metrics))
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
    ax.text(0.58, y - 0.028, m["minilm_val"], fontsize=11.0, fontweight='bold', 
            color='#0F172A', ha='center', va='center')
    if m["badge_minilm"]:
        badge_col = '#16A34A' if m["winner"] == "minilm" else ('#D97706' if 'ชนะ' in m["badge_minilm"] else '#64748B')
        ax.text(0.58, y - 0.054, m["badge_minilm"], fontsize=8.5, fontweight='bold',
                color=badge_col, ha='center', va='center')

    # Wangchan Box (Column 2)
    wang_bg = '#EFF6FF' if m["winner"] != "wangchan" else '#DCFCE7'
    wang_border = '#93C5FD' if m["winner"] != "wangchan" else '#22C55E'
    rect_w = patches.FancyBboxPatch((0.72, y - box_height + 0.022), 0.22, box_height - 0.015,
                                    boxstyle="round,pad=0.008,rounding_size=0.01",
                                    linewidth=1.5, edgecolor=wang_border, facecolor=wang_bg)
    ax.add_patch(rect_w)
    ax.text(0.83, y - 0.028, m["wangchan_val"], fontsize=11.0, fontweight='bold', 
            color='#1E40AF' if m["winner"] != "wangchan" else '#15803D', ha='center', va='center')
    if m["badge_wangchan"]:
        badge_col = '#15803D' if m["winner"] == "wangchan" else '#D97706'
        ax.text(0.83, y - 0.054, m["badge_wangchan"], fontsize=8.5, fontweight='bold',
                color=badge_col, ha='center', va='center')

# Column Headers
ax.text(0.58, 0.865, 'MiniLM-L12 (Multilingual)', fontsize=12, fontweight='bold', color='#475569', ha='center')
ax.text(0.83, 0.865, 'WangchanBERTa (โมเดลที่เราเลือก)', fontsize=12, fontweight='bold', color='#1D4ED8', ha='center')

# Bottom Summary banner
summary_rect = patches.FancyBboxPatch((0.04, 0.010), 0.92, 0.045,
                                      boxstyle="round,pad=0.008,rounding_size=0.01",
                                      linewidth=1.2, edgecolor='#3B82F6', facecolor='#EFF6FF')
ax.add_patch(summary_rect)
ax.text(0.5, 0.032, 
        'ข้อสรุปเชิงวิศวกรรม: MiniLM ดึงติดใน 5 อันดับแรกได้ดีกว่า (93.3% vs 80%) แต่คำตอบไปกองอยู่อันดับล่าง (Hit@1 แค่ 40%)\nWangchan แม่นยำอันดับ 1 มากกว่า (53.3%), แตกคำไทยดีกว่า และเมื่อเสริม BM25 คะแนน Hit@5 พุ่งขึ้นมาแตะ 93.3% เท่ากัน!',
        fontsize=9.8, fontweight='bold', color='#1E40AF', ha='center', va='center')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
output_path = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart1_wangchan_vs_minilm_head_to_head.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated: {output_path}")
