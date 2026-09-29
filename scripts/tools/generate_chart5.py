# -*- coding: utf-8 -*-
"""
Script to generate Chart 5: Comprehensive & Fair Local LLM Benchmark Comparison
Apple-to-Apple Comparison in the ~3B Tier:
- Qwen2.5:3B (3.1B) vs Llama3.2:3B (3.2B) vs Gemma2:2B (2.6B) vs Llama3.2:1B (1.2B) vs Qwen2.5:0.5B (0.5B)
- Scientific Truth: Testing Hallucination Vulnerability, Prompt Adherence, Area Code Drift, and Inference Speed.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import json

# Setup Thai font
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False

ASSETS_DIR = Path("data/presentation_assets")
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Load benchmark results
json_file = Path("data/exp4_llm_benchmark_results.json")
if not json_file.exists():
    json_file = Path("finalproject/data/exp4_llm_benchmark_results.json")

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

models = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b", "llama3.2:1b", "qwen2.5:0.5b"]
display_names = ["Qwen2.5:3B\n(Selected)", "Llama3.2:3B\n(Meta Peer)", "Gemma2:2B\n(Google)", "Llama3.2:1B\n(Meta Edge)", "Qwen2.5:0.5B\n(Ultra-Light)"]

# Scientific Scores based on the 5-case test suite:
# Both 3B models followed Thai instructions well, but ALL small models struggled with unanswerable zero-shot queries
# Speed: Qwen 14.8 tok/s vs Llama3.2:3B 9.3 tok/s (Qwen is 59% faster on same GPU!)
speed_vals = [data[m]["tokens_per_sec"] for m in models]
vram_vals = [data[m]["vram_gb"] for m in models]
prompt_adhere_vals = [100.0, 100.0, 80.0, 100.0, 60.0]

fig = plt.figure(figsize=(16, 11))
fig.suptitle("การทดลองที่ 4: การเปรียบเทียบโมเดลภาษา Local LLMs แบบเป็นกลางในระดับ 3B (Fair Benchmark)\nพิสูจน์ข้อจำกัดการหลอน (Hallucination Vulnerability) และความเร็วในการประมวลผลบน GPU จริง",
             fontsize=15, fontweight="bold", y=0.98)

# Grid layout: 2 rows, row 1 = 2 bar charts, row 2 = table & qualitative evidence
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.22)

# Subplot 1: Speed (Tokens/sec) comparison - Qwen vs Llama 3B shootout
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(models))
w = 0.55
bars = ax1.bar(x, speed_vals, w, color=["#2ECC71", "#E67E22", "#3498DB", "#9B59B6", "#95A5A6"], edgecolor="black")

for b in bars:
    h = b.get_height()
    ax1.annotate(f"{h:.1f} tok/s", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                 textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=10)

ax1.set_title("ความเร็วในการสังเคราะห์คำตอบจริงบน GPU (Inference Speed: Tokens/sec)", fontsize=12, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(display_names, fontsize=9.5)
ax1.set_ylim(0, 30)
ax1.set_ylabel("Tokens / วินาที (ยิ่งสูงยิ่งตอบไว)", fontsize=11)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Annotate Qwen 3B vs Llama 3B speed gap
ax1.annotate('Qwen 3B เร็วกว่า Llama 3B ถึง 59%\n(14.8 vs 9.3 tok/s) บน GPU เดียวกัน',
             xy=(0, 15.5), xytext=(0.4, 21.0),
             arrowprops=dict(facecolor='#27AE60', shrink=0.08, width=2, headwidth=7),
             fontsize=9.5, fontweight='bold', color='#196F3D',
             bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5", ec="#27AE60", lw=1.2))

# Subplot 2: VRAM (GB) & Prompt Adherence
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()

r1 = ax2.bar(x - 0.18, vram_vals, 0.35, label="VRAM Usage (GB) [แกนซ้าย]", color="#34495E", edgecolor="black")
r2 = ax2_twin.bar(x + 0.18, prompt_adhere_vals, 0.35, label="Prompt Adherence (%) [แกนขวา]", color="#1ABC9C", edgecolor="black")

for r in r1:
    h = r.get_height()
    ax2.annotate(f"{h:.1f}G", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                 textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9.5)
for r in r2:
    h = r.get_height()
    ax2_twin.annotate(f"{int(h)}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                      textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9.5)

ax2.set_title("การใช้หน่วยความจำ VRAM (GB) และการปฏิบัติตามคำสั่งของพรอมต์ (%)", fontsize=12, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(display_names, fontsize=9.5)
ax2.set_ylim(0, 3.2)
ax2_twin.set_ylim(0, 125)
ax2.set_ylabel("VRAM (GB)", fontsize=11, color="#2C3E50")
ax2_twin.set_ylabel("Prompt Adherence (%)", fontsize=11, color="#16A085")
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Combine legends
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9.5)

# Subplot 3 (Bottom Span): Unbiased Hallucination Qualitative Table
ax_table = fig.add_subplot(gs[1, :])
ax_table.axis('off')

# Table data demonstrating transparent empirical facts
table_data = [
    ["โมเดล (Model)", "พารามิเตอร์", "ผลลัพธ์คำถามที่ไม่มีในบริบท (ถามหาเบอร์โทรบ้านนครใน)", "การวิเคราะห์การหลอน (Hallucination Analysis)", "เหตุผลทางวิศวกรรมที่เลือก Qwen2.5:3B"],
    ["Qwen2.5:3B\n(Selected)", "3.1B", '"เบอร์ติดต่อบ้านนครใน 074-227555"\n(อีกรอบ: "ไม่ได้ให้บริการ แต่เข้าชมฟรี")', "หลอนตามรหัสพื้นที่จริง (สงขลา = 074)\nยังคงมีแนวโน้มกุเบอร์เมื่อถูกถามตรงๆ", "1. ความเร็วสูงสุดในรุ่น 3B (14.8 tok/s)\n2. เข้าใจบริบทไทยและรหัสพื้นที่สงขลา\n3. คุมง่ายด้วย Guardrails ของระบบ RAG"],
    ["Llama3.2:3B\n(Meta Peer)", "3.2B", '"เบอร์โทรศัพท์สำหรับติดต่อของบ้านนครในคือ 077-221-111"', "หลอนข้ามจังหวัดไปยังสุราษฎร์ฯ (077)\nกุเบอร์โทรศัพท์เช่นกัน", "1. ตอบสนองช้ากว่า 59% (เหลือเพียง 9.3 tok/s)\n2. ไม่คุ้นเคยกับภูมิศาสตร์เฉพาะถิ่นของภาคใต้ตอนล่าง"],
    ["Gemma2:2B", "2.6B", '"บ้านนครใน: 08-1881-1111"\n(และตอบข้อเท็จจริงผิดว่า: "ไม่ correct")', "หลอนกุเบอร์มือถือปลอม\nและสลับไปใช้ภาษาอังกฤษ", "หลุดไวยากรณ์ไทย และความเร็วค่อนข้างช้า (10.5 tok/s)"],
    ["Llama3.2:1B", "1.2B", '"เบอร์โทรศัพท์ติดต่อของบ้านนครใน ... คือ 02- 282 8888"', "หลอนข้ามภาคไปยังกรุงเทพฯ (02)\nทั้งที่สถานที่จริงอยู่ในสงขลา", "ขนาดพารามิเตอร์ 1.2B เล็กเกินไป ขาดความรู้ภูมิศาสตร์ไทย"],
    ["Qwen2.5:0.5B", "0.5B", '"ขอเบอร์โทรศัพท์ติดต่อของบ้านนครใน: 031-1234567"\n(และดึงร้านอาหารคาวมาปนของหวาน)', "หลอนรหัสทางไกลที่ไม่มีจริง (031)\nและไม่ผ่านเงื่อนไข Negative Constraint", "พารามิเตอร์ 0.5B ไม่สามารถทำตามคำสั่งที่มีข้อห้ามซับซ้อนได้"]
]

tab = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                     bbox=[0.01, 0.02, 0.98, 0.92],
                     colWidths=[0.13, 0.08, 0.31, 0.25, 0.23])
tab.auto_set_font_size(False)
tab.set_fontsize(9.5)

# Style table
for (row, col), cell in tab.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1A252F')
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 1:
        cell.set_facecolor('#E8F8F5') # Soft green highlight for Qwen2.5:3B
        cell.set_text_props(color='#0E6655', fontweight='bold' if col in [0, 4] else 'normal')
    elif row == 2:
        cell.set_facecolor('#FEF9E7') # Soft yellow for Llama 3B peer
    else:
        cell.set_facecolor('#F8F9F9' if row % 2 == 1 else '#FFFFFF')

ax_table.set_title("ตารางหลักฐานเชิงประจักษ์แบบโปร่งใส (Transparent Scientific Evidence Table)",
                   fontsize=12.5, fontweight="bold", pad=12)

plt.tight_layout(rect=[0, 0, 1, 0.97])
chart5_path = ASSETS_DIR / "chart5_llm_benchmark_comparison.png"
plt.savefig(chart5_path, dpi=300)
plt.close()
print(f"[OK] Saved {chart5_path}")
