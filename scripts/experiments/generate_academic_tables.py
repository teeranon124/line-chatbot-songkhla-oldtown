import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set clean academic font styling
plt.rcParams['font.sans-serif'] = ['Tahoma', 'Leelawadee UI', 'Angsana New', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def create_table1():
    # Table 1: Embedding Benchmark
    fig, ax = plt.subplots(figsize=(13, 5), facecolor='#FFFFFF')
    ax.axis('off')
    
    fig.text(0.5, 0.92, 'ตารางที่ 1: การเปรียบเทียบคุณสมบัติและประสิทธิภาพของโมเดล Embedding', 
             ha='center', fontsize=16, weight='bold', color='#1E3A8A')
    fig.text(0.5, 0.85, 'วัดผลบนชุดคำเฉพาะภาษาไทยเมืองเก่าสงขลา 15 คำ และทดสอบความเร็วบน CPU (Batch=15)', 
             ha='center', fontsize=11, color='#475569')

    columns = [
        'ชื่อโมเดล Embedding', 'ขนาดโมเดล\n(Parameters)', 'การแตกคำเฉพาะ\n(Token Frag.)', 
        'CPU Latency\n(Batch=15)', 'Standalone Dense MRR\n(ค้นหาเวกเตอร์เดี่ยว)', 'บทวิเคราะห์และการตัดสินใจทางวิศวกรรม'
    ]
    
    data = [
        ['MiniLM-L12\n(paraphrase-multilingual)', '117.7M', '5.2 tokens', '76.4 ms', '0.706', 'เร็วและเบาที่สุด แต่ความแม่นยำภาษาไทยต่ำสุด (MRR 0.706)'],
        ['mE5-Base (Prefix)\n(intfloat/multilingual-e5)', '278.0M', '5.2 tokens', '200.9 ms', '0.917 (สูงสุด)', 'แม่นยำที่สุดในโหมดเดี่ยว แต่โมเดลใหญ่ถึง 278M และทำงานช้าสุด'],
        ['ConGen WangchanBERTa\n(โมเดลที่เราคัดเลือก)', '105.8M\n(เบาสุด)', '4.4 tokens\n(ตัดคำดีสุด)', '164.8 ms', '0.894\n(ตามหลัง 0.023)', 'เบากว่า mE5 2.6 เท่า ตัดคำเฉพาะไทยดีสุด และชดเชยจุดอ่อนด้วย BM25 ได้']
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.4)

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
            if j == 0 or j == 5:
                cell.set_text_props(ha='left')
            if i == 2:
                cell.set_text_props(weight='bold' if j in [1, 2, 5] else 'normal')
                if j == 0:
                    cell.set_text_props(color='#1E3A8A', weight='bold')

    plt.tight_layout()
    out = r'c:\social\A_krit2\finalproject\data\presentation_assets\table1_embedding_benchmark.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated Table 1')

def create_table2():
    # Table 2: Retrieval Ablation
    fig, ax = plt.subplots(figsize=(14, 5.5), facecolor='#FFFFFF')
    ax.axis('off')
    
    fig.text(0.5, 0.92, 'ตารางที่ 2: ผลการทดลอง Ablation Study สถาปัตยกรรมค้นหา 4 รูปแบบ', 
             ha='center', fontsize=16, weight='bold', color='#1E3A8A')
    fig.text(0.5, 0.85, 'วัดผลบนชุดคำถามทดสอบ 15 ข้อ (5 Exact Entities, 5 Semantic QA, 5 Spatial Multi-hop)', 
             ha='center', fontsize=11, color='#475569')

    columns = [
        'สถาปัตยกรรมการค้นหา', 'Exact Names\n(5 ข้อ) MRR', 'Semantic QA\n(5 ข้อ) MRR', 
        'Spatial Multi-hop\n(5 ข้อ) MRR', 'Overall Hit@1', 'Overall Hit@3', 'Overall MRR', 'บทสรุปและจุดเด่น-จุดด้อย'
    ]
    
    data = [
        ['1. BM25 Only (Lexical)', '1.000', '0.833', '0.400', '66.7%', '80.0%', '0.744', 'แม่นชื่อเฉพาะ 100% แต่ตอบคำถามเชิงพื้นที่หลายต่อไม่ได้'],
        ['2. Dense Only (FAISS)', '0.600', '0.933', '0.600', '60.0%', '80.0%', '0.711', 'เข้าใจภาษาธรรมชาติได้ดี แต่ตกม้าตายเมื่อเจอชื่อเฉพาะ'],
        ['3. Graph Only (NetworkX)', '0.467', '0.200', '0.900', '46.7%', '60.0%', '0.522', 'แม่นความเชื่อมโยงของพื้นที่ แต่ไม่มีความยืดหยุ่นทางข้อความ'],
        ['4. Hybrid GraphRAG (RRF)', '1.000', '0.933', '0.933', '93.3%', '100.0%', '0.956', 'สมบูรณ์แบบที่สุด ผสานจุดแข็งและปิดจุดบอดของทุกระบบ 100%']
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.3)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='#FFFFFF', weight='bold')
        cell.set_edgecolor('#CBD5E1')

    for i in range(len(data)):
        bg = '#F8FAFC' if i % 2 == 0 else '#FFFFFF'
        if i == 3:
            bg = '#EFF6FF'
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#CBD5E1')
            if j == 0 or j == 7:
                cell.set_text_props(ha='left')
            if i == 3:
                cell.set_text_props(weight='bold')
                if j in [4, 5, 6]:
                    cell.set_text_props(color='#1E3A8A', weight='bold')

    plt.tight_layout()
    out = r'c:\social\A_krit2\finalproject\data\presentation_assets\table2_retrieval_ablation.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated Table 2')

def create_table3():
    # Table 3: Local LLM Shootout
    fig, ax = plt.subplots(figsize=(13.5, 6), facecolor='#FFFFFF')
    ax.axis('off')
    
    fig.text(0.5, 0.92, 'ตารางที่ 3: ผลการประเมิน Local LLMs ขนาด 1B - 3B บน GPU ตัวเดียวกัน', 
             ha='center', fontsize=16, weight='bold', color='#1E3A8A')
    fig.text(0.5, 0.85, 'วัด Throughput (tokens/s), อัตราใช้ VRAM, คุณภาพภาษาไทย และรูปแบบความหลอนเมื่อไม่มี Guardrail', 
             ha='center', fontsize=11, color='#475569')

    columns = [
        'โมเดลที่ทดสอบ', 'ขนาดตัวแปร\n(Parameters)', 'VRAM Footprint\n(กิกะไบต์)', 
        'Throughput\n(ความเร็ว)', 'คุณภาพภาษาไทย\n(Fluency)', 'พฤติกรรมความหลอน (เมื่อยังไม่คุมด้วย Guardrail)'
    ]
    
    data = [
        ['Qwen2.5:3B\n(โมเดลที่เราเลือก)', '3.1B', '1.9 GB', '14.8 tok/s\n(เร็วกว่า 3B ตัวอื่น 59%)', 'ดีเยี่ยม\n(สละสลวย เป็นธรรมชาติ)', 'หลอนเบอร์โทรโดยเดารหัสพื้นที่สงขลา (074)'],
        ['Llama3.2:3B', '3.2B', '2.0 GB', '9.3 tok/s', 'ปานกลาง\n(ติดโครงสร้างแปลอังกฤษ)', 'หลอนเบอร์โทรโดยเดารหัสสุราษฎร์ธานี (077)'],
        ['Gemma2:2B', '2.6B', '1.6 GB', '11.2 tok/s', 'ปานกลาง', 'สับสนชื่อร้านและตำแหน่งถนน'],
        ['SmolLM2:1.7B', '1.7B', '1.8 GB', '12.5 tok/s', 'ต่ำ\n(ไวยากรณ์ผิดบ่อย)', 'เกิดลูปข้อความซ้ำ (Repetition Loop) ข้อมูลตกหล่น'],
        ['Llama3.2:1B', '1.2B', '1.3 GB', '18.4 tok/s', 'ต่ำ\n(ตัดทอนประโยคสั้น)', 'หลอนเบอร์โทรโดยเดารหัสกรุงเทพฯ (02)']
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='#FFFFFF', weight='bold')
        cell.set_edgecolor('#CBD5E1')

    for i in range(len(data)):
        bg = '#F8FAFC' if i % 2 == 0 else '#FFFFFF'
        if i == 0:
            bg = '#EFF6FF'
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#CBD5E1')
            if j == 0 or j == 5:
                cell.set_text_props(ha='left')
            if i == 0:
                cell.set_text_props(weight='bold')
                if j in [0, 3]:
                    cell.set_text_props(color='#1E3A8A', weight='bold')

    plt.tight_layout()
    out = r'c:\social\A_krit2\finalproject\data\presentation_assets\table3_llm_benchmark.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated Table 3')

def create_table4_and_chart6():
    # Clean Academic Version of Chart 6 & Table 4
    fig, ax = plt.subplots(figsize=(14.5, 7.5), facecolor='#FFFFFF')
    ax.axis('off')
    
    fig.text(0.5, 0.94, 'ตารางที่ 4 / Chart 6: ผลการทดสอบจริง Universal Anti-Hallucination Guardrail Engine', 
             ha='center', fontsize=16, weight='bold', color='#1E3A8A')
    fig.text(0.5, 0.88, 'หลักฐานเชิงประจักษ์ Before vs After ข้าม 4 มิติสำคัญ เพื่อควบคุมโมเดลขนาด 3B ให้ตอบตรง Ground Truth 100%', 
             ha='center', fontsize=11, color='#475569')

    columns = [
        'มิติที่ควบคุม', 'ตัวอย่างคำถามทดสอบ', 'ผลลัพธ์ของโมเดลปกติ (Before / หลอน)', 
        'ผลลัพธ์หลังผ่าน Guardrail (After / ควบคุม)', 'กลไกที่ใช้ตรวจสอบ (Mechanism)'
    ]
    
    data = [
        [
            '1. เบอร์โทรศัพท์\n(Phone Numbers)', 
            '"ขอเบอร์โทรศัพท์ติดต่อ\nของบ้านนครในหน่อยครับ"', 
            'เบอร์ติดต่อบ้านนครใน คือ 074-321-456\n(หลอนตัวเลขรหัสพื้นที่ 074 ขึ้นมาเอง)', 
            '"ขออภัยครับ ในฐานข้อมูลยังไม่มีการระบุ\nเบอร์โทรศัพท์ติดต่อของสถานที่ดังกล่าว"', 
            'Regex Digit Grounding\n(ดักจับและเทียบกับบริบท)'
        ],
        [
            '2. เวลาเปิด-ปิด\n(Operating Hours)', 
            '"โรงสีแดง หับ โห้ หิ้น\nเปิดทำการกี่โมงถึงกี่โมง"', 
            'เปิดทุกวันเวลา 08:30 - 17:00 น.\n(หลอนเวลามาตรฐานออฟฟิศ ทั้งที่ไม่มีในข้อมูล)', 
            '"ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุ\nเวลาเปิด-ปิดที่แน่นอน แนะนำตรวจสอบกับสถานที่"', 
            'Time Format Verification\n(สกัดตัวเลขเวลาเดา)'
        ],
        [
            '3. ราคา / ค่าเข้าชม\n(Pricing & Fees)', 
            '"บ้านนครใน\nค่าเข้าชมกี่บาท"', 
            'ค่าเข้าชมผู้ใหญ่ 50 บาท เด็ก 20 บาท\n(หลอนตัวเลขค่าบัตรขึ้นมาเอง)', 
            '"ขออภัยครับ ในฐานข้อมูลยังไม่ได้ระบุ\nราคาหรือค่าเข้าชมที่แน่ชัด แนะนำสอบถามหน้าร้าน"', 
            'Price Token Grounding\n(ตรวจสอบตัวเลขราคา/บาท)'
        ],
        [
            '4. ตำแหน่งถนน\n(Spatial Consistency)', 
            '"ร้านแต้เฮี้ยงอิ๋ว\nตั้งอยู่ถนนสายใด"', 
            'ร้านแต้เฮี้ยงอิ๋ว ตั้งอยู่บน ถนนนครนอก\n(โมเดลสับสนตำแหน่งถนน)', 
            '"ร้านแต้เฮี้ยงอิ๋ว ตั้งอยู่บน ถนนนางงาม..."\n(แก้กลับมาเป็นถนนที่ถูกต้องตามจริง)', 
            'Knowledge Graph Topology\n(ดึง Ground Truth จากกราฟ)'
        ]
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 2.7)

    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor('#1E3A8A')
        cell.set_text_props(color='#FFFFFF', weight='bold')
        cell.set_edgecolor('#CBD5E1')

    for i in range(len(data)):
        bg = '#F8FAFC' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(columns)):
            cell = table[i+1, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor('#CBD5E1')
            if j in [0, 1, 2, 3]:
                cell.set_text_props(ha='left')
            if j == 2:
                cell.set_text_props(color='#DC2626') # Red for hallucinated
            elif j == 3:
                cell.set_text_props(color='#15803D', weight='bold') # Green for safe
            elif j == 4:
                cell.set_text_props(color='#1E3A8A', weight='bold')

    plt.tight_layout()
    # Save both as table4 and update chart6
    out1 = r'c:\social\A_krit2\finalproject\data\presentation_assets\table4_guardrail_verification.png'
    out2 = r'c:\social\A_krit2\finalproject\data\presentation_assets\chart6_guardrail_verification_proof.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print('Generated Table 4 and updated Chart 6')

if __name__ == '__main__':
    create_table1()
    create_table2()
    create_table3()
    create_table4_and_chart6()
    print('All academic table images generated!')
