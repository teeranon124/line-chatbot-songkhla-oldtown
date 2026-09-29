# Phase B Result

PARTIAL

> Baseline นี้ไม่ปรับ retrieval, fusion, Top-K, prompt หรือโมเดลใด ๆ และไม่ดาวน์โหลดโมเดลใหม่

## Environment

- Branch: `achi`
- Python: `3.12.14`
- Current Local LLM: `qwen2.5:3b`
- Embedding model: `kornwtp/ConGen-model-wangchanberta`
- Graph backend: NetworkX pickle (Neo4j ใช้เมื่อ local endpoint เข้าถึงได้)
- Dataset: 50 ข้อ (48 answerable, 2 insufficient)

## Current Hybrid Definition

Dense + BM25 + Graph → Weighted RRF (`k=60`, weights `0.45/0.30/0.25`) และใช้ Dynamic Top-K เดิม (`2/4/7`)

## Dataset Validation

**PASS**

| Category | Count |
|---|---:|
| comparison | 5 |
| dense_friendly | 3 |
| direct_fact | 8 |
| entity_relation | 8 |
| graph_friendly | 5 |
| insufficient_evidence | 2 |
| lexical_mismatch | 5 |
| multi_hop_relation | 5 |
| natural_thai | 4 |
| recommendation | 5 |

## Retrieval Results

| Mode | Hit@1 | Hit@3 | Hit@5 | MRR | Mean latency (ms) | Status |
|---|---:|---:|---:|---:|---:|---|
| dense | — | — | — | — | — | NOT_RUN |
| sparse | — | — | — | — | — | NOT_RUN |
| graph | 0.646 | 0.688 | 0.688 | 0.667 | 0.084 | OK |
| hybrid | — | — | — | — | — | NOT_RUN |

Hit ใช้ exact ID ตาม modality; ไม่ใช้ fuzzy title/content matching. Insufficient-evidence ถูกตัดออกจาก retrieval metrics.

## Category Results (Hit@3)

| Category | Dense | Sparse | Graph | Hybrid |
|---|---:|---:|---:|---:|
| direct_fact | — | — | 0.875 | — |
| entity_relation | — | — | 1.000 | — |
| multi_hop_relation | — | — | 1.000 | — |
| comparison | — | — | 1.000 | — |
| recommendation | — | — | 0.000 | — |
| natural_thai | — | — | 0.250 | — |
| lexical_mismatch | — | — | 0.000 | — |
| graph_friendly | — | — | 1.000 | — |
| dense_friendly | — | — | 0.667 | — |

## Answer-Level Results

**NOT RUN — OLLAMA UNAVAILABLE**

ไม่มีการสลับไปใช้ Groq หรือโมเดลอื่น และไม่มีการสร้างคะแนนคำตอบเทียม

Subjective dimensions: **HUMAN_REVIEW_REQUIRED** — Correctness, Context Faithfulness, Completeness, Relevance, Thai Naturalness, Recommendation Usefulness

## Graph Value Evidence / Hybrid Failure Cases

ไม่สามารถสรุป strong Hybrid success หรือ failure/neutral cases ได้ เพราะ Dense และ Hybrid ไม่ได้รันครบ การเลือกตัวอย่างจาก Graph/Sparse เพียงสองโหมดจะไม่ตอบคำถามการทดลองและเสี่ยงทำให้ข้อสรุปเอนเอียง

## Phase A Validation

Actual Hybrid prompt validation: **NOT RUN** เพราะ current Dense model และ current Local LLM ไม่พร้อมใช้งานในสภาพแวดล้อมนี้

Regression boundary test ยังคงตรวจว่า `graph_evidence` ถูก render เป็น `[หลักฐานกราฟ]` ใน prompt แต่ไม่ถูกนับแทนการรันจริง

## Latency

รายงานเฉพาะ warm retrieval latency ของโหมดที่รันได้; generation/total latency ไม่มีเพราะ answer-level ไม่ได้รัน

## Failure Analysis

- Dense: ไม่ได้รัน — embedding model ที่ตั้งค่าไว้ไม่มีใน local cache และห้ามดาวน์โหลด
- Graph: ดูผลจริงใน JSON; การจับคู่ entity เป็น lexical จึงอาจพลาดคำบรรยายที่ไม่เอ่ยชื่อโหนด
- Hybrid: ไม่ได้รันเพราะ Dense component เริ่มต้นไม่ได้
- LLM: Ollama/current configured model ไม่พร้อม จึงไม่ประเมิน generation

## Main Conclusion

1. Hybrid outperform Dense overall? **ตอบไม่ได้ — ทั้งสองโหมดไม่ได้รัน**
2. Hybrid outperform Graph overall? **ตอบไม่ได้ — Hybrid ไม่ได้รัน**
3. Graph ช่วยหมวดใดมากสุด? ในผลที่รันได้ Hit@3 = 1.000 ที่ entity relation, multi-hop, comparison และ graph-friendly แต่ยังเปรียบเทียบกับ Dense ไม่ได้
4. Hybrid ช่วยหมวดใดมากสุด? **ตอบไม่ได้**
5. Hybrid ทำให้แย่ลงที่ใด? **ตอบไม่ได้**
6. มีหลักฐานพอว่า Hybrid ดีกว่าหรือไม่? **ไม่มี**
7. Phase C ควร optimize อะไรก่อน? **ยังไม่ควร optimize; ต้องทำ baseline บน environment เดิมให้ครบก่อน**

## Phase C Recommendation

ยังไม่ควรเริ่ม optimization. ขั้นแรกคือทำให้ environment เดิมพร้อมด้วย configured ConGen model และ configured local LLM แล้ว rerun baseline เดิมโดยไม่เปลี่ยน dataset/metric definition

## Blockers

- sparse: Current BM25 index could not be loaded: ModuleNotFoundError: No module named 'rank_bm25'
- dense: Configured embedding model 'kornwtp/ConGen-model-wangchanberta' is not present in the local cache; Phase B forbids downloading or switching models.
- hybrid: Configured embedding model 'kornwtp/ConGen-model-wangchanberta' is not present in the local cache; Phase B forbids downloading or switching models.
- answer: NOT RUN — OLLAMA UNAVAILABLE

## Regression Tests

- Before benchmark: `8 passed, 0 failed, 0 errors`
- After benchmark: `8 passed, 0 failed, 0 errors`

## Files Created

- `evaluation/phase_b_dataset.json`
- `evaluation/phase_b_runner.py`
- `evaluation/results/phase_b/phase_b_validation.json`
- `evaluation/results/phase_b/phase_b_retrieval_results.json`
- `evaluation/results/phase_b/phase_b_answer_results.json`
- `evaluation/results/phase_b/phase_b_summary.csv`
- `evaluation/results/phase_b/phase_b_category_summary.csv`
- `evaluation/results/phase_b/phase_b_report.md`

## Files Modified

ไม่มีไฟล์เดิมถูกแก้ไข

## Git Safety

- Initial branch: `achi`
- Final branch: `achi`
- Main modified: NO
- Commit created: NO
- Push performed: NO

## Final Repository Status

มีเฉพาะไฟล์ใหม่ภายใต้ `evaluation/`; ดู `git status` จาก final verification

