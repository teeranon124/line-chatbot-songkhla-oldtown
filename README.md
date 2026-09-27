# 🏛️ line-chatbot-songkhla-oldtown
### ระบบผู้ช่วยอัจฉริยะนำเที่ยวย่านเมืองเก่าสงขลา ด้วย Hybrid GraphRAG และ Dual LLM บน LINE Official Account

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![LINE Bot](https://img.shields.io/badge/LINE-Messaging%20API-00B900.svg)](https://developers.line.biz/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.com/)
[![Groq](https://img.shields.io/badge/Groq-Cloud%20LLM-orange.svg)](https://groq.com/)
[![FAISS](https://img.shields.io/badge/FAISS-Dense%20Search-yellowgreen.svg)](https://github.com/facebookresearch/faiss)
[![NetworkX](https://img.shields.io/badge/NetworkX-Knowledge%20Graph-blueviolet.svg)](https://networkx.org/)

โครงงานปลายภาควิชา **241-351 AI for Social Media**  
ภาควิชาวิศวกรรมคอมพิวเตอร์ คณะวิศวกรรมศาสตร์ มหาวิทยาลัยสงขลานครินทร์ (PSU)  
**คณะผู้พัฒนา**: นาย ธีรนนท์ ทองสีดำ (รหัสนักศึกษา: 6710110196) และคณะ

---

## 🌟 จุดเด่นของระบบ (Key Features)

1. **100% Rich Interactive LINE Flex Messages & Carousels**:
   - ไม่มีการตอบด้วย Plain text ทื่อๆ ทุกคำตอบถูกจัดวางในรูปแบบการ์ด **AI Smart Card** พร้อมป้ายบอกสถานะการประมวลผล (`💻 Local` / `⚡ Cloud`)
   - **Multi-Entity Recognition**: เมื่อถามถึงหลายสถานที่ (เช่น *"ร้านไอติมโอ่ง กับร้านเจ๊นิ เปิดกี่โมง"*) บอทจะตอบการ์ดสรุป พร้อม**แนบคารูเซลการ์ดของ 2 ร้านนั้นตรงเผง**ในข้อความเดียว
   - มีปุ่ม Action แบบโต้ตอบทันที: `📍 แผนที่นำทาง` (Google Maps URL) และ `📞 โทรออก` (Direct Dialer)

2. **True Hybrid Retrieval Architecture (Tri-Retrieval + RRF)**:
   - **Dense Retrieval**: `kornwtp/ConGen-model-wangchanberta` (768 มิติ L2-normalized) บน FAISS FlatIP
   - **Sparse Retrieval**: `BM25Okapi` ตัดคำภาษาไทยด้วย PyThaiNLP (`newmm`)
   - **Knowledge Graph Retrieval**: โครงข่ายความสัมพันธ์ Ontology (9 Classes / 7 Relations) ผ่าน NetworkX DiGraph และ Neo4j Cypher
   - **Reciprocal Rank Fusion (RRF)**: ผสานคะแนนอันดับด้วย $k=60$ ถ่วงน้ำหนักแบบปรับได้

3. **Dual LLM Architecture with Adaptive Model Router**:
   - **Local LLM (Default)**: Ollama (`qwen2.5:3b`) ทำงานแบบ Offline-first ประหยัด รวดเร็ว ไร้ค่าใช้จ่าย สำหรับตอบข้อเท็จจริง เวลาเปิด-ปิด พิกัด และเมนูเด่น
   - **Cloud API LLM**: Groq (`qwen/qwen3.8-27b`) ประมวลผลเฉพาะงานสังเคราะห์แผนการเดินทางหลายวัน หรือการคิดวิเคราะห์เชิงลึก
   - **Automatic Failover**: หากเครื่อง Local ไม่ตอบสนอง ระบบจะสลับไปเรียก Cloud API อัตโนมัติทันที

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
├── finalproject/
│   ├── data/                 # ข้อมูลดิบ, Chunks, ดัชนี FAISS/BM25, และ Knowledge Graph
│   │   ├── songkhla_places_facts.json        # ฐานข้อมูลข้อเท็จจริง 22 สถานที่
│   │   ├── songkhla_rag_chunks.json          # ข้อมูล Chunks สำหรับค้นคืน
│   │   ├── songkhla_faiss.index              # ดัชนีเวกเตอร์ FAISS
│   │   ├── songkhla_bm25.pkl                 # ดัชนีคำค้น BM25
│   │   ├── songkhla_graph.pkl                # กราฟความรู้ NetworkX DiGraph
│   │   └── songkhla_knowledge_graph.html     # Interactive PyVis HTML Graph
│   ├── scripts/              # 7-Step Sequential Pipeline
│   │   ├── 01_enrich_places.py               # รวบรวมข้อมูลสถานที่
│   │   ├── 02_extract_and_chunk.py           # สกัดและทำ Semantic Chunking
│   │   ├── 03_extract_ontology_triples.py    # สกัด RDF Triples & Schema
│   │   ├── 04_build_graph.py                 # สร้าง Knowledge Graph
│   │   ├── 05_build_indices.py               # สร้างดัชนี FAISS + BM25
│   │   ├── 06_benchmark_evaluation.py        # รันการทดสอบและประเมินผล
│   │   ├── 07_test_end_to_end.py             # ทดสอบระบบและ LINE Schema Validation
│   │   ├── run_pipeline.py                   # Master Pipeline Orchestrator
│   │   └── PIPELINE_GUIDE.md                 # คู่มือการรันไปป์ไลน์ฉบับละเอียด
│   ├── src/                  # ซอร์สโค้ดระบบหลัก
│   │   ├── config.py                         # การตั้งค่าระบบและตัวแปรสภาพแวดล้อม
│   │   ├── flex_templates.py                 # เทมเพลต LINE Flex Cards & Carousels
│   │   ├── graph_engine.py                   # ตัวสืบค้นและท่องกราฟความรู้
│   │   ├── intents.py                        # ตัวจัดหมวดหมู่เจตนา (Intent Classifier)
│   │   ├── line_handler.py                   # ตัวจัดการข้อความและการตอบกลับ LINE
│   │   ├── rag_engine.py                     # ตัวขับเคลื่อน Dual LLM & Prompt Pipeline
│   │   ├── retriever.py                      # ตัวค้นหาแบบไฮบริด (Dense + Sparse + Graph)
│   │   └── webhook.py                        # Flask Webhook Server
│   └── static/images/        # รูปภาพสถานที่จริงความละเอียดสูง
├── run_pipeline.bat          # สคริปต์รันไปป์ไลน์แบบคลิกเดียวบน Windows
├── requirements.txt          # รายการ Library dependencies
├── .env.example              # ตัวอย่างไฟล์ตั้งค่าคีย์ API
└── README.md
```

---

## 🚀 คู่มือการติดตั้งสำหรับผู้ร่วมพัฒนา (Setup Guide for Teammates)

### 1. โคลน Repository
```bash
git clone https://github.com/teeranon124/line-chatbot-songkhla-oldtown.git
cd line-chatbot-songkhla-oldtown
```

### 2. สร้างและเปิดใช้งาน Virtual Environment
```bash
python -m venv venv
# สำหรับ Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# หรือ Command Prompt:
.\venv\Scripts\activate.bat
```

### 3. ติดตั้ง Library Dependencies
```bash
pip install -r requirements.txt
```

### 4. ตั้งค่า Environment Variables
คัดลอกไฟล์ `.env.example` ไปเป็น `.env` และใส่คีย์ของคุณ:
```bash
copy .env.example .env
copy .env.example finalproject\.env
```
กำหนดค่าใน `.env`:
* `LINE_CHANNEL_SECRET`: Secret จาก LINE Developers Console
* `LINE_CHANNEL_ACCESS_TOKEN`: Token จาก LINE Developers Console
* `GROQ_API_KEY`: API Key จาก [Groq Console](https://console.groq.com/)

### 5. ติดตั้ง Local LLM (Ollama)
ดาวน์โหลด [Ollama](https://ollama.com/) แล้วรันคำสั่งดาวน์โหลดโมเดล:
```bash
ollama pull qwen2.5:3b
```

---

## 🎛️ การทดสอบและรันระบบ (Running the Project)

### รันตรวจสอบระบบครบวงจร (End-to-End Verification)
```bash
.\run_pipeline.bat --verify
```

### รัน Webhook Server เพื่อเชื่อมต่อ LINE
```bash
python finalproject/src/webhook.py
```

### เปิด Cloudflare Tunnel เชื่อมต่อเซิร์ฟเวอร์กับ LINE Webhook
```bash
cloudflared tunnel --url http://localhost:5000
```
นำ URL ที่ได้ (เช่น `https://xxxx.trycloudflare.com/callback`) ไปวางที่ช่อง **Webhook URL** บน LINE Developers Console แล้วกด **Verify** และเปิดใช้งาน **Use Webhook**
