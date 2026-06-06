# Eval Overview

Total cases: **20**

## BERTScore
Model: `bert-base-multilingual-cased`

## Average Metrics
| Metric | Value |
| --- | ---: |
| rag_context_relevance | 0.6095 |
| rag_groundedness | 0.9373 |
| rag_answer_relevance | 1.0 |
| rag_triad_mean | 0.8489 |
| bertscore_precision | 1.0 |
| bertscore_recall | 1.0 |
| bertscore_f1 | 1.0 |
| answer_expected_term_coverage | 0.9667 |
| context_expected_term_coverage | 0.8758 |

## Category Breakdown
| Category | Count | RAG triad mean | Answer term coverage |
| --- | ---: | ---: | ---: |
| conversation | 1 | 0.6781 | 0.3333 |
| definition | 2 | 0.8706 | 1.0 |
| law_rule | 4 | 0.86 | 1.0 |
| licensing | 1 | 0.8631 | 1.0 |
| penalty_car | 7 | 0.8558 | 1.0 |
| penalty_motorbike | 3 | 0.8726 | 1.0 |
| remedial_measure | 1 | 0.7615 | 1.0 |
| scope | 1 | 0.886 | 1.0 |

## Lowest RAG Triad Cases
- `followup_memory_age` (conversation): 0.6781 - Tôi 17 tuổi thì đã đủ tuổi thi bằng lái ô tô chưa?
- `remedial_measure_sign` (remedial_measure): 0.7615 - Một biện pháp khắc phục hậu quả trong lĩnh vực giao thông đường bộ là gì?
- `car_alcohol_over_80` (penalty_car): 0.813 - Người điều khiển ô tô có nồng độ cồn vượt quá 80 mg/100 ml máu bị phạt bao nhiêu?
- `car_parking_sidewalk` (penalty_car): 0.8202 - Đỗ xe ô tô trên hè phố trái quy định bị phạt bao nhiêu?
- `car_horn_22_5` (penalty_car): 0.8371 - Người điều khiển ô tô bấm còi trong khu đông dân cư từ 22h đến 5h bị phạt bao nhiêu?
