# SMARTSpan Dataset

SMARTSpan originates from a randomized controlled trial (RCT) on cardiovascular disease prevention focused on improving statin adherence and healthy behavior change. It comprises 173 anonymized health coaching (HC) notes and supports two tasks:

1) **Goal Extraction** — find all behavior-change goal spans per HC note.  
2) **SMARTness Classification** — label each extracted goal as Specific (S), Measurable (M), and Attainable (A), then derive a final class: SMART / Partially SMART / Not SMART.

### Cross-validation

As the original dataset is relativly small, we created 5 folds using 70/15/15 splits per fold (123 train, 25 val, 25 test notes).

### Goal extraction

Two annotators reviewed all 173 HC notes and highlighted any text that stated a behavioral goal. Extracted goals per HC note ranged from **0 to 5**; notably, **27%** of HC notes had **no goals (46/173)** and **27%** had **exactly one (46/173)**. This sparsity contrasts with MultiSpanQA, where most examples contain multiple answer spans. As shown in Table 1, the number of extracted goals per test split varies from 24 to 47.

<sup><b>Table 1.</b>  Distribution of the number of goals per HC note and the total number of goals in the SMARTSpan <em>test</em> sets across five data splits.</sup>
| goals per HC note | 0  | 1 | 2 | 3 | 4 | 5 | Σ Goals |
|-------------------|----|---|---|---|---|---|--------:|
| Split_1           | 5  | 9 | 3 | 7 | 0 | 1 |     41 |
| Split_2           | 4  | 7 | 4 | 8 | 2 | 0 |     47 |
| Split_3           | 4  | 7 | 8 | 2 | 4 | 0 |     45 |
| Split_4           | 7  | 6 | 5 | 4 | 2 | 1 |     41 |
| Split_5           | 12 | 5 | 5 | 3 | 0 | 0 |     24 |

### SMARTness classification

Each extracted goal is annotated for three components—Specific (S), Measurable (M), and Attainable (A). Relevant (R) is omitted due to its subjectivity and Time-bound (T) is treated as implicit (all goals are reviewed by the next HC session). Each component is binary (0/1). The final label is derived deterministically: SMART if S=1,M=1,A=1; Partially SMART if exactly two are 1; Not SMART otherwise.

<sup><b>Table 2.</b> SMARTness label distribution in the SMARTSpan <em>test</em> sets across five data splits.</sup>
| SMARTness classification  | SMART   | Partially SMART | Not SMART | Σ Goals |
|-------------------------- |-------- |-----------------|-----------|--------:|
|Split_1	                  |18	      |13	              |10	        |41        |
|Split_2	                  |20	      |13	              |14	        |47        |
|Split_3	                  |22	      |11	              |12	        |45        |
|Split_4	                  |21	      |11	              |9	        |41        |
|Split_5	                  |10	      |9	              |5	        |24        |
