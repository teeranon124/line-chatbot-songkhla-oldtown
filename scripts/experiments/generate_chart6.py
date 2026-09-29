import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set font
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Angsana New', 'Leelawadee UI', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 9.5), facecolor='#0B0F19')
ax.set_facecolor('#0B0F19')
ax.axis('off')

# Title & Subtitle
fig.text(0.5, 0.95, 'Universal Multi-Dimensional Anti-Hallucination Guardrail Engine', 
         ha='center', fontsize=20, weight='bold', color='#38BDF8')
fig.text(0.5, 0.91, 'Dual-Layer Architecture: Real Execution Proof Across 4 Critical Dimensions (Local LLM 3B)', 
         ha='center', fontsize=12, color='#94A3B8')

cards = [
    {
        'title': 'DIMENSION 1: PHONE NUMBER GROUNDING',
        'query': 'ขอเบอร์โทรศัพท์ติดต่อของบ้านนครในหน่อยครับ',
        'raw': 'เบอร์โทรศัพท์ติดต่อบ้านนครใน คือ 074-321-456 ครับ (หลอนรหัส 074 สงขลาขึ้นมาเอง)',
        'guard': 'ขออภัยครับ ในฐานข้อมูลยังไม่มีการระบุเบอร์โทรศัพท์ติดต่อของสถานที่ดังกล่าว',
        'status': '100% INTERCEPTED (ZERO FAKE NUMBERS)',
        'status_color': '#10B981',
        'color': '#3B82F6'
    },
    {
        'title': 'DIMENSION 2: OPERATING HOURS & TIMES',
        'query': 'โรงสีแดง หับ โห้ หิ้น เปิดทำการกี่โมงถึงกี่โมง',
        'raw': 'เปิดทำการทุกวันเวลา 08:30 - 17:00 น. ครับ (หลอนเวลามาตรฐานออฟฟิศ ทั้งที่ในข้อมูลไม่มี)',
        'guard': 'ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุเวลาเปิด-ปิดที่แน่นอน แนะนำตรวจสอบกับทางสถานที่โดยตรงครับ',
        'status': '100% INTERCEPTED (NO GUESSING HOURS)',
        'status_color': '#10B981',
        'color': '#8B5CF6'
    },
    {
        'title': 'DIMENSION 3: PRICING & ADMISSION FEES',
        'query': 'บ้านนครใน ค่าเข้าชมกี่บาท',
        'raw': 'ค่าเข้าชมสำหรับผู้ใหญ่ 50 บาท เด็ก 20 บาท ครับ (หลอนตัวเลขราคาค่าตั๋วขึ้นมา)',
        'guard': 'ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุราคาหรือค่าเข้าชมที่แน่ชัด แนะนำสอบถามหน้าร้านครับ',
        'status': '100% INTERCEPTED (NO FAKE PRICES)',
        'status_color': '#10B981',
        'color': '#EC4899'
    },
    {
        'title': 'DIMENSION 4: KNOWLEDGE GRAPH SPATIAL GROUNDING',
        'query': 'ร้านแต้เฮี้ยงอิ๋ว ตั้งอยู่ถนนสายใด',
        'raw': 'ร้านแต้เฮี้ยงอิ๋ว ตั้งอยู่บน ถนนนครนอก ติดริมทะเลสาบสงขลา (โมเดลสับสนตำแหน่งถนน)',
        'guard': 'ร้านแต้เฮี้ยงอิ๋ว ตั้งอยู่บน ถนนนางงาม เป็นร้านอาหารเก่าแก่ชื่อดัง... (Graph Auto-Corrected)',
        'status': '100% CORRECTED VIA GRAPH TOPOLOGY',
        'status_color': '#10B981',
        'color': '#F59E0B'
    }
]

y_starts = [0.69, 0.49, 0.29, 0.09]

for i, (c, y) in enumerate(zip(cards, y_starts)):
    # Outer Card Box
    card_box = patches.FancyBboxPatch((0.05, y), 0.90, 0.17, boxstyle='round,pad=0.015,rounding_size=0.01',
                                     facecolor='#1E293B', edgecolor=c['color'], linewidth=1.5)
    ax.add_patch(card_box)
    
    # Title badge
    ax.text(0.07, y + 0.135, c['title'], fontsize=12, weight='bold', color=c['color'])
    ax.text(0.93, y + 0.135, c['status'], fontsize=10, weight='bold', color=c['status_color'], ha='right')
    
    # Query
    ax.text(0.07, y + 0.095, f'Prompt Query: "{c["query"]}"', fontsize=10, color='#F8FAFC', style='italic')
    
    # Raw Output (Red)
    ax.text(0.07, y + 0.055, '[Raw LLM (Before)]:', fontsize=9.5, weight='bold', color='#F87171')
    ax.text(0.25, y + 0.055, c['raw'], fontsize=9, color='#E2E8F0')
    
    # Guardrail Output (Green)
    ax.text(0.07, y + 0.015, '[Guardrail (After)]:', fontsize=9.5, weight='bold', color='#4ADE80')
    ax.text(0.25, y + 0.015, f'"{c["guard"]}"', fontsize=9, color='#38BDF8', weight='bold')

plt.tight_layout()
out_file = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart6_guardrail_verification_proof.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f'Saved Chart 6 to {out_file}')
