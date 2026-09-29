# Phase B.1 Result

**PASS**

Baseline นี้ใช้ dataset และ production configuration เดิมทั้งหมด ไม่มีการ optimize หรือแก้ production code

# Frozen Dataset

- Path: `evaluation/phase_b_dataset.json`
- SHA-256: `4e1101d07e5f3d1ee49c22ba87a7d984179b3af8dfb37936242a944330526e76`
- Questions: 50
- Answerable: 48
- Insufficient evidence: 2
- Category distribution: comparison=5, dense_friendly=3, direct_fact=8, entity_relation=8, graph_friendly=5, insufficient_evidence=2, lexical_mismatch=5, multi_hop_relation=5, natural_thai=4, recommendation=5
- Dataset modified during benchmark: NO

# Environment

- Python: 3.12.10 (`.venv`)
- Embedding: `kornwtp/ConGen-model-wangchanberta`, 768 dimensions
- FAISS: existing index, 21 vectors, dimension 768
- Sparse: `rank_bm25.BM25Okapi`
- Graph backend: NetworkX in-memory
- Graph size: 74 nodes / 98 edges
- Neo4j required: NO
- Hybrid definition: Dense + Sparse/BM25 + Graph → Weighted RRF
- Fusion weights: Dense 0.45, Sparse 0.30, Graph 0.25
- RRF k: 60
- Dynamic Top-K: min=2, default=4, max=7
- Ollama: `http://localhost:11434`
- Local LLM: `qwen2.5:3b`, temperature=0.2, top_p=0.9, seed not configured

# Phase A Regression

- Before: 8/8 PASS
- After: 8/8 PASS

# Retrieval Results

Initialization was excluded. Each mode received one unmeasured warm-up; latency wraps production `retrieve()` only.

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Mean Latency | Median Latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.562 | 0.771 | 0.875 | 0.675 | 26.598 ms | 13.873 ms |
| Sparse | 0.542 | 0.875 | 0.938 | 0.710 | 0.240 ms | 0.238 ms |
| Graph | 0.646 | 0.688 | 0.688 | 0.667 | 0.119 ms | 0.115 ms |
| Hybrid | 0.729 | 0.854 | 0.875 | 0.790 | 14.562 ms | 13.980 ms |

Relevance uses exact frozen IDs: Dense/Sparse use `relevant_chunk_ids`; Graph uses `relevant_graph_ids`; Hybrid uses their union. The two insufficient-evidence items are excluded. Precision/Recall/nDCG are omitted because graph nodes and text chunks are different, non-exhaustively judged relevance units.

# Category Results

ตารางใช้ Hit@3; `n` คือจำนวน answerable ในหมวด

| Category | n | Dense | Sparse | Graph | Hybrid |
|---|---:|---:|---:|---:|---:|
| comparison | 5 | 0.800 | 1.000 | 1.000 | 1.000 |
| dense_friendly | 3 | 1.000 | 1.000 | 0.667 | 1.000 |
| direct_fact | 8 | 0.750 | 1.000 | 0.875 | 0.875 |
| entity_relation | 8 | 0.750 | 0.875 | 1.000 | 0.875 |
| graph_friendly | 5 | 1.000 | 0.800 | 1.000 | 1.000 |
| insufficient_evidence | 2 | N/A | N/A | N/A | N/A |
| lexical_mismatch | 5 | 0.800 | 0.800 | 0.000 | 0.600 |
| multi_hop_relation | 5 | 0.600 | 0.800 | 1.000 | 1.000 |
| natural_thai | 4 | 0.500 | 0.500 | 0.250 | 0.500 |
| recommendation | 5 | 0.800 | 1.000 | 0.000 | 0.800 |

# Retrieval Winner

- Best overall (balanced Hit@1/MRR): Hybrid
- Best Hit@1: Hybrid (0.729)
- Best Hit@3: Sparse (0.875)
- Best Hit@5: Sparse (0.938)
- Best MRR: Hybrid (0.790)
- Dataset มีเพียง 48 answerable questions; ความต่างเหล่านี้ยังไม่ควรถูกอ้างว่าเป็นสากลหรือมีนัยสำคัญทางสถิติ

# Answer-Level Results

Automatic Correctness คือ acceptable-phrase hit อย่างน้อยหนึ่งคำจาก frozen ground truth; Groundedness คือ phrase เดียวกันปรากฏทั้งในคำตอบและ rendered context ทั้งสองค่าเป็นเพียง proxy ไม่ใช่ semantic judgment

| Mode | Automatic Correctness | Groundedness Proxy | Refusal | Mean Generation Latency | Mean Total Latency |
|---|---:|---:|---:|---:|---:|
| Dense | 64.6% | 64.6% | 50.0% (2 ข้อ) | 5119.4 ms | 5145.6 ms |
| Sparse | 64.6% | 64.6% | 50.0% (2 ข้อ) | 4235.5 ms | 4235.8 ms |
| Graph | 68.8% | 62.5% | 100.0% (2 ข้อ) | 1642.2 ms | 1642.5 ms |
| Hybrid | 60.4% | 60.4% | 50.0% (2 ข้อ) | 4609.6 ms | 4624.3 ms |

Subjective Thai quality: **HUMAN_REVIEW_REQUIRED**

# Phase A Real Prompt Validation

Status: **PASS** — ตรวจจาก prompt ที่ส่งจริงหลัง production `build_context()`; ไม่เปิดเผย system prompt

## PB017

- Question: มีร้านอาหารหรือขนมใดอยู่ถนนเดียวกับร้านไอติมโอ่งบ้าง?
- Graph evidence: `graph_aitim_oang` — โครงสร้างความสัมพันธ์ใน Knowledge Graph สำหรับ: ร้านไอติมโอ่ง - ประเภท: FoodShop (ของหวาน / เครื่องดื่ม) - ถนนที่ตั้ง: ถนนนางงาม - เวลาเปิด-ปิด: 10:00 - 18:30 น. - ช่วงราคา: 20 - 30 บาท - คะแนนรีวิว: 4.3 ดาว (1250 รีวิว) - โครงข่ายความสัมพันธ์เชื่อมโยงออกไป: L…
- Evidence present in final prompt: YES
- Answer grounded in that evidence: UNCERTAIN

## PB018

- Question: สถานที่ใดบ้างอยู่ถนนนครนอกสายเดียวกับโรงสีแดง หับโห้หิ้น?
- Graph evidence: `graph_songkhla_station` — โครงสร้างความสัมพันธ์ใน Knowledge Graph สำหรับ: สงขลาสเตชั่น - ประเภท: FoodShop (ของหวาน / คาเฟ่และศิลปะ) - ถนนที่ตั้ง: ถนนนครนอก - เวลาเปิด-ปิด: จันทร์-ศุกร์ 09:30 - 18:00 น. / เสาร์-อาทิตย์ 08:00 - 19:00 น. - ช่วงราคา: 50 - 120 บาท - คะแนนรีวิว: 4.4 ดาว (430…
- Evidence present in final prompt: YES
- Answer grounded in that evidence: YES

## PB019

- Question: จากโรงแรมสงขลาแต่แรกเดินไปโรงสีแดงและร้านไอติมโอ่งประมาณกี่เมตร?
- Graph evidence: `graph_aitim_oang` — โครงสร้างความสัมพันธ์ใน Knowledge Graph สำหรับ: ร้านไอติมโอ่ง - ประเภท: FoodShop (ของหวาน / เครื่องดื่ม) - ถนนที่ตั้ง: ถนนนางงาม - เวลาเปิด-ปิด: 10:00 - 18:30 น. - ช่วงราคา: 20 - 30 บาท - คะแนนรีวิว: 4.3 ดาว (1250 รีวิว) - โครงข่ายความสัมพันธ์เชื่อมโยงออกไป: L…
- Evidence present in final prompt: YES
- Answer grounded in that evidence: UNCERTAIN

# Graph Helps

ตัวอย่างตามกฎวัดล่วงหน้า (Graph hit, Hybrid preserves/improves rank หรือ answer phrase hit):

- PB021: Dense rank=3, Graph rank=1, Hybrid rank=1; Dense answer hit=False, Hybrid answer hit=True
- PB023: Dense rank=1, Graph rank=1, Hybrid rank=1; Dense answer hit=False, Hybrid answer hit=True
- PB018: Dense rank=4, Graph rank=1, Hybrid rank=1; Dense answer hit=True, Hybrid answer hit=True

# Graph Hurts / Neutral

กรณีที่ Hybrid แย่กว่า Dense; เป็น association ที่วัดได้ ไม่ใช่ข้อพิสูจน์เชิงเหตุผลว่า Graph เป็นสาเหตุทั้งหมด:

- PB019: Dense rank=1, Graph rank=2, Hybrid rank=3; Dense answer hit=True, Hybrid answer hit=False
- PB040: Dense rank=1, Graph rank=None, Hybrid rank=1; Dense answer hit=True, Hybrid answer hit=False
- PB045: Dense rank=1, Graph rank=1, Hybrid rank=1; Dense answer hit=True, Hybrid answer hit=False

Neutral cases ตามกฎเดียวกัน: 18 ข้อ

# Failure Analysis

- Dense: ranking failure=2 (เช่น PB003, PB010); semantic miss=4 (เช่น PB028, PB033, PB034, PB037)
- Sparse: ranking failure=2 (เช่น PB010, PB044); vocabulary mismatch=1 (เช่น PB036)
- Graph: entity matching issue=12 (เช่น PB007, PB027, PB028, PB031, PB033); graph traversal/ontology limitation=3 (เช่น PB029, PB030, PB046)
- Hybrid: fusion ranking or Dynamic Top-K limitation=6 (เช่น PB003, PB010, PB028, PB033, PB034)
- LLM: expected phrase absent (human review required)=68; refusal error=3. Phrase absence ต้องตรวจโดยมนุษย์ก่อนสรุปว่าเนื้อหาผิด

# Baseline Verdict

1. Hybrid beat Dense overall? **YES สำหรับ Hit@1, Hit@3 และ MRR; Hit@5 เสมอ**
2. Hybrid beat Sparse overall? **MIXED — Hybrid ชนะ Hit@1/MRR แต่ Sparse ชนะ Hit@3/Hit@5**
3. Hybrid beat Graph overall? **YES ใน retrieval metrics ทั้งสี่ค่า**
4. Graph เด่นสุดใน entity_relation, multi_hop_relation, comparison และ graph_friendly (Hit@3=1.000)
5. Dense เด่นใน dense_friendly และ graph_friendly (Hit@3=1.000) และยังแข็งแรงใน lexical mismatch/recommendation (0.800)
6. Sparse เด่นใน direct_fact, comparison, recommendation และ dense_friendly (Hit@3=1.000)
7. Hybrid เด่นใน multi_hop, comparison, graph_friendly และ dense_friendly (Hit@3=1.000)
8. Hybrid regress เทียบ Sparse ใน direct_fact/recommendation/lexical_mismatch และมี measured hurt cases 5 ข้อ
9. Graph evidence improves actual answers? **มีตัวอย่างเฉพาะราย แต่ไม่ดีขึ้นโดยรวม** — Hybrid phrase hit 60.4%, ต่ำกว่า Dense 64.6% และ Graph 68.8%; ต้อง human review ก่อนสรุป semantic quality
10. Enough evidence to begin Phase C? **YES สำหรับเริ่ม optimization แบบ controlled** โดยต้องคง dataset/hash/metrics นี้เป็น baseline

# Phase C Recommendation

1. Fusion ranking/reranking และ context selection — Hybrid แพ้ Sparse ที่ Hit@3/5 และมี 5 measured regressions
2. Graph entity matching/query routing — Graph พลาด entity matching 12 ข้อและได้ 0 ใน recommendation/lexical mismatch
3. Dynamic Top-K/context budget — Hybrid exact-ID miss 6 ข้อ และบาง simple queries คืนเพียง k=2 จน evidence ถูกเบียดออก

ยังไม่ได้ implement การปรับใด ๆ

# Files Created

- `evaluation/phase_b_complete_runner.py`
- `evaluation/phase_b_complete_report.py`
- `evaluation/results/phase_b_complete/phase_b_complete_retrieval_results.json`
- `evaluation/results/phase_b_complete/phase_b_complete_answer_results.json`
- `evaluation/results/phase_b_complete/phase_b_complete_summary.csv`
- `evaluation/results/phase_b_complete/phase_b_complete_category_summary.csv`
- `evaluation/results/phase_b_complete/phase_b_complete_failure_analysis.json`
- `evaluation/results/phase_b_complete/phase_b_complete_prompt_validation.json`
- `evaluation/results/phase_b_complete/phase_b_complete_run_manifest.json`
- `evaluation/results/phase_b_complete/phase_b_complete_report.md`

# Files Modified

Production files: **NONE**

# Git Safety

- Branch: `achi`
- Main modified: NO
- Production code modified: NO
- Commit created: NO
- Push performed: NO

# Final Recommendation

**READY FOR PHASE C**

Baseline ครบทั้ง retrieval, answer generation, real prompt boundary, latency และ failure analysis แล้ว โดยต้องใช้ dataset SHA-256 เดิมเป็น control ใน Phase C
