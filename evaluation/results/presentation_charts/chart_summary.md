# Presentation Chart Summary

## retrieval_methods_comparison.png

- Exact sources: `evaluation/results/phase_b_complete/phase_b_complete_retrieval_results.json` (`overall`) and `evaluation/results/phase_b_complete/phase_b_complete_summary.csv`; duplicated values were checked for exact numeric agreement.
- Metrics: Hit@1, Hit@3, Hit@5, and MRR on 48 answerable questions.
- Key finding: Hybrid leads Hit@1 (0.729) and Hybrid leads MRR (0.790); Sparse leads Hit@3 (0.875) and Sparse leads Hit@5 (0.938). Hybrid does not win every metric.
- คำอธิบายสำหรับนำเสนอ: Hybrid เด่นด้านการจัดอันดับผลลัพธ์แรกและ MRR ขณะที่ Sparse ครอบคลุมผลลัพธ์ในอันดับ 3 และ 5 ได้ดีกว่า จึงควรอธิบายว่าแต่ละวิธีมีจุดแข็งต่างกัน

## hybrid_optimization_comparison.png

- Exact sources: `evaluation/results/final_optimization/final_optimization_results.csv` and `evaluation/results/final_optimization/final_optimization_report.md`.
- Metrics: Hit@1, Hit@3, Hit@5, and MRR for Baseline and configurations A–F; RRF k=60 for every configuration.
- Key finding: Configuration D has higher retrieval metrics than Baseline (MRR 0.821 vs 0.790), but the recorded final decision is **KEEP BASELINE** because answer-level refusal behavior became worse.
- คำอธิบายสำหรับนำเสนอ: แม้น้ำหนักชุด D จะเพิ่มคะแนน retrieval แต่ผลยืนยันระดับคำตอบทำให้ความสามารถในการปฏิเสธคำถามที่ไม่มีหลักฐานลดลง จึงคง baseline 0.45/0.30/0.25

## category_performance.png

- Exact sources: `evaluation/results/phase_b_complete/phase_b_complete_category_summary.csv` and `evaluation/results/phase_b_complete/phase_b_complete_retrieval_results.json` (`by_category`); duplicated values were checked for exact numeric agreement.
- Metric: Hit@3 for nine answerable categories. `insufficient_evidence` is excluded because retrieval relevance is N/A.
- Key finding: Graph reaches 1.000 in entity relation, multi-hop relation, comparison, and graph-friendly categories, but scores 0.000 in recommendation and lexical mismatch. Sparse reaches 1.000 in direct fact, comparison, recommendation, and dense-friendly categories.
- คำอธิบายสำหรับนำเสนอ: กราฟช่วยคำถามเชิงความสัมพันธ์ได้ชัดเจน แต่ไม่เหมาะกับทุกหมวด ส่วน Sparse แข็งแรงกับคำถามข้อเท็จจริงและคำแนะนำ จึงเป็นเหตุผลที่ระบบผสมหลาย retriever

## answer_quality_comparison.png

- Exact source: `evaluation/results/phase_b_complete/phase_b_complete_answer_results.json` (`overall` and `automatic_metric_limitations`), cross-checked with `evaluation/results/phase_b_complete/phase_b_complete_report.md`.
- Metrics: acceptable-phrase hit, evidence-supported phrase hit, and refusal rate on two insufficient-evidence questions.
- Key finding: These are deterministic string-match proxies, not semantic correctness or human-rated quality. Graph has the highest acceptable-phrase proxy (68.8%) and refusal rate (100.0%); Hybrid records 60.4% and 50.0%.
- คำอธิบายสำหรับนำเสนอ: ตัวเลขนี้ใช้ตรวจคำหรือวลีที่กำหนดไว้เท่านั้น จึงใช้เป็นสัญญาณประกอบและยังต้องให้มนุษย์ประเมินคุณภาพภาษาไทยและความถูกต้องเชิงความหมาย

## latency_comparison.png

- Exact sources: `evaluation/results/phase_b_complete/phase_b_complete_retrieval_results.json` (`overall`) and `evaluation/results/phase_b_complete/phase_b_complete_summary.csv`.
- Metrics: mean and median production `retrieve()` latency after one unmeasured warm-up; initialization excluded. A logarithmic x-axis is used because measured values span more than two orders of magnitude.
- Key finding: Graph and Sparse retrieval are sub-millisecond in this run, while Hybrid mean retrieval latency is 14.562 ms and Dense is 26.598 ms.
- คำอธิบายสำหรับนำเสนอ: เวลา retrieval ของทุกวิธียังต่ำเมื่อเทียบกับเวลาสร้างคำตอบ และแกน logarithmic ช่วยให้เห็นค่าที่ต่างกันมากโดยไม่ซ่อน Graph/Sparse

## answer_latency_comparison.png

- Exact source: `evaluation/results/phase_b_complete/phase_b_complete_answer_results.json` (`overall`).
- Metrics: mean LLM generation latency and mean end-to-end answer latency; warm-up excluded.
- Key finding: Answer generation dominates total latency. Graph context has the lowest mean total latency (1642.5 ms), while Dense has the highest (5145.6 ms).
- คำอธิบายสำหรับนำเสนอ: คอขวดหลักอยู่ที่การสร้างคำตอบของ LLM ไม่ใช่ retrieval จึงควรแยกสองสเกลนี้ออกจากกันในการนำเสนอ

## Overall findings

- Strength: Hybrid provides the best Hit@1 and MRR in the production baseline, while Sparse provides the best Hit@3 and Hit@5.
- Strength: Graph is particularly effective for relation-oriented categories.
- Weakness: No single retrieval method dominates every metric or category; Graph is weak for recommendation and lexical mismatch in this dataset.
- Weakness: Answer metrics are proxy checks and subjective Thai answer quality remains `HUMAN_REVIEW_REQUIRED`.
- Decision: **KEEP BASELINE** with Dense=0.45, Sparse=0.30, Graph=0.25, and RRF k=60.
