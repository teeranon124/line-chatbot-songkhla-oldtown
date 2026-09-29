import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Render Futsal-style Table 1: BERT Only vs BERT + BM25
fig, ax = plt.subplots(figsize=(16, 5.8), facecolor='#FFFFFF')
ax.axis('off')

fig.text(0.5, 0.93, 'ตารางเปรียบเทียบโมเดล BERT: ระหว่าง "Dense Only" กับ "BERT + BM25"', 
         ha='center', fontsize=16, weight='bold', color='#1E3A8A')
fig.text(0.5, 0.86, 'พิสูจน์การเพิ่มเทคนิค BM25 เพื่อยกระดับโมเดลภาษาไทย (WangchanBERTa) ให้สู้ Multilingual ได้โดยคงความเบาของระบบ', 
         ha='center', fontsize=11, color='#475569')

columns = [
    'โมเดล BERT ที่ทดสอบ', 'ขนาดโมเดล\n(Parameters)', 'เวกเตอร์เพียวๆ\n(Dense Only MRR)', 
    'เวกเตอร์ + BM25\n(Hybrid MRR)', 'การพัฒนา\n(Delta MRR)', 'Hit@1\n(+BM25)', 'ข้อสรุปทางวิศวกรรม (Engineering Takeaway)'
]

data = [
    [
        'MiniLM-L12\n(Multilingual เล็ก)', 
        '117.7M', 
        '0.6002\n(ต่ำสุด)', 
        '0.7852', 
        '+0.1850\n(+30.8%)', 
        '66.7%', 
        'เมื่อมี BM25 ช่วยดึงคำเฉพาะ ทำให้คะแนนพุ่งขึ้นชัดเจน'
    ],
    [
        'mE5-Base (Prefix)\n(Multilingual ใหญ่)', 
        '278.0M\n(หนักสุด)', 
        '0.9667\n(สูงสุด)', 
        '0.9222', 
        '-0.0444\n(-4.6%)', 
        '86.7%', 
        'เวกเตอร์ดีอยู่แล้ว แต่โมเดลใหญ่ถึง 278M หนักเครื่องและกิน RAM มาก'
    ],
    [
        'ConGen WangchanBERTa\n(โมเดลที่เราเลือก)', 
        '105.8M\n(เบาที่สุด)', 
        '0.6730\n(ตามหลัง mE5)', 
        '0.7796\n(สู้ MiniLM ได้)', 
        '+0.1066\n(+15.8%)', 
        '66.7%\n(เท่า MiniLM)', 
        'เดิมเวกเตอร์ตามหลัง แต่พอ + BM25\nคะแนนพุ่งขึ้นมาสูสีทันที แถมเบากว่า mE5 2.62 เท่า'
    ]
]

table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center', colWidths=[0.16, 0.11, 0.12, 0.12, 0.11, 0.09, 0.29])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
table.scale(1, 2.5)

# Style Header
for j in range(len(columns)):
    cell = table[0, j]
    cell.set_facecolor('#1E3A8A')
    cell.set_text_props(color='#FFFFFF', weight='bold')
    cell.set_edgecolor('#CBD5E1')

# Style Rows
for i in range(len(data)):
    bg = '#F8FAFC' if i % 2 == 0 else '#FFFFFF'
    if i == 2: # Highlight Wangchan
        bg = '#EFF6FF'
    for j in range(len(columns)):
        cell = table[i+1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#CBD5E1')
        if j == 0 or j == 6:
            cell.set_text_props(ha='left')
        if i == 2:
            cell.set_text_props(weight='bold' if j in [1, 3, 4, 6] else 'normal')
            if j == 0:
                cell.set_text_props(color='#1E3A8A', weight='bold')
            elif j == 4:
                cell.set_text_props(color='#15803D', weight='bold')

plt.tight_layout()
out = r'c:\social\A_krit2\finalproject\data\presentation_assets\table1_bert_vs_bert_bm25.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f'Generated Futsal-style Table 1 to {out}')
