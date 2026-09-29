# Final Hybrid Optimization

Dataset SHA-256: `4e1101d07e5f3d1ee49c22ba87a7d984179b3af8dfb37936242a944330526e76` (unchanged)

## BASELINE

- Weights: 0.45 / 0.30 / 0.25
- Hit@1: 0.729
- Hit@3: 0.854
- Hit@5: 0.875
- MRR: 0.790
- Answer proxy: acceptable=64.6%, grounded=64.6%, refusal=100.0%, mean total=3868.0 ms

## WINNER

- Configuration: D
- Weights: 0.30 / 0.40 / 0.30
- Tie note: F เท่ากับ D ทุก selection metric; เลือก D ตามลำดับ configuration ที่กำหนดไว้เพื่อทำ answer confirmation เพียงชุดเดียว
- Hit@1: 0.750
- Hit@3: 0.896
- Hit@5: 0.917
- MRR: 0.821
- Answer proxy: acceptable=70.8%, grounded=70.8%, refusal=50.0%, mean total=2975.2 ms

## DELTA (Winner - Baseline)

- Hit@1: +0.021
- Hit@3: +0.042
- Hit@5: +0.042
- MRR: +0.031
- Acceptable phrase hit: +6.2%
- Groundedness proxy: +6.2%
- Refusal: -50.0%

## CATEGORY SAFETY CHECK (Hit@3)

| Category | n | Baseline | Winner | Delta |
|---|---:|---:|---:|---:|
| entity_relation | 8 | 0.875 | 1.000 | +0.125 |
| multi_hop_relation | 5 | 1.000 | 1.000 | +0.000 |
| recommendation | 5 | 0.800 | 0.800 | +0.000 |
| lexical_mismatch | 5 | 0.600 | 0.600 | +0.000 |
| graph_friendly | 5 | 1.000 | 1.000 | +0.000 |

Category regressions:
- ไม่มี Hit@3 regression ใน 5 หมวดที่กำหนด

## DECISION

**KEEP BASELINE**

- Retrieval improvement criterion: True
- Severe graph-oriented regression: False
- Answer proxy clearly worse: True

## FINAL RECOMMENDED HYBRID

- Dense: 0.45
- Sparse: 0.30
- Graph: 0.25
- RRF k: 60

Subjective quality: **HUMAN_REVIEW_REQUIRED**

Phase A tests: PASS (8/8)
Production code modified: NO
Commit created: NO
Push performed: NO
