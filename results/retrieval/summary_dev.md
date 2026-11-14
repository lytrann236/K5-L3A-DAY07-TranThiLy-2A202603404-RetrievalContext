| Config | Method | top-k | Budget (chars) | Neighbors | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Avg context (chars / chunks) | Median ms/query | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| M1_dense | dense | 3 | 2400 | 0 | 9/15 (60%) | 13/15 (87%) | 13/15 (87%) | 1940 / 3 | 13.6 | 0 USD |
| M2_bm25 | bm25 | 3 | 2400 | 0 | 11/15 (73%) | 13/15 (87%) | 13/15 (87%) | 1888 / 3 | 0.1 | 0 USD |
| M3_hybrid | hybrid | 3 | 2400 | 0 | 10/15 (67%) | 14/15 (93%) | 14/15 (93%) | 1937 / 3 | 13.6 | 0 USD |
| I1_hybrid_rerank | hybrid+rerank | 3 | 2400 | 0 | 14/15 (93%) | 14/15 (93%) | 14/15 (93%) | 1816 / 2.82 | 93.0 | 0 USD |
| I2_neighbors | hybrid+rerank (based on I1_hybrid_rerank) | 3 | 4000 | 1 | 14/15 (93%) | 14/15 (93%) | 14/15 (93%) | 3537 / 5.82 | 92.0 | 0 USD |
| I3_rerank_maxseg | hybrid+rerank_maxseg | 3 | 2400 | 0 | 13/15 (87%) | 15/15 (100%) | 15/15 (100%) | 1985 / 3 | 186.6 | 0 USD |
