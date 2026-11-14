# Code Day 7 — cách chạy

Corpus: OpenStax *Introduction to Python Programming*, Ch.8 Strings + Ch.9 Lists (CC BY-NC-SA 4.0), dùng lại từ Day 6
(repo `K5-L3A-DAY06-TranThiLy-2A202603404-DataIngestion`): 92 chunk của index B, PostgreSQL 16 + pgvector 0.8.7
(Docker `day6_student_starter`), embedding `BAAI/bge-small-en-v1.5` (384 chiều).

## Cài đặt
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# .env (không commit): GEMINI_API_KEY=..., LLM_MODEL=gemini-3.5-flash-lite,
#                      DATABASE_URL=postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab
# Docker Day 6 phải đang chạy (container day6_student_starter)
python3 -m unittest discover -s tests          # 27 test offline (mock chỉ kiểm tra code)
```

## Pipeline
| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Xuất dữ liệu Day 6 | `python3 src/export_day6.py --day6-questions <Day6>/benchmark/questions.jsonl` | `data/day7/{chunks,questions,cases}.json` |
| 2. Retrieval + context (không gọi LLM) | `python3 src/eval_retrieval.py --split dev` (I2: `--configs I2_neighbors --best I1_hybrid_rerank`) | `results/retrieval/<config>_dev/`, `summary_dev.md` |
| 3. Agent trên dev | `python3 src/run_agent_bench.py --retrieval <config> --split dev` | `results/agent/<config>_dev/{traces,summary}.json` |
| 4. Chốt cấu hình | `configs/selection_criteria.md`, `configs/selected.json` (commit 7668985 trước test) | |
| 5. Test (1 lần) | `python3 src/eval_retrieval.py --split test --configs M1_dense I3_rerank_maxseg`; `run_agent_bench.py --retrieval {M1_dense,I3_rerank_maxseg} --split test` | `results/*/..._test/` |
| 6. Chấm tay | `python3 src/label_agent.py --config <config> --split test` | `results/agent/<config>_test/reviews.json` |
| 7. Tổng hợp | `python3 starter/agent_metrics.py --cases data/day7/cases.json --traces .../traces.json --reviews .../reviews.json --split test --output .../metrics.json` | `metrics.json` |

## File
- `common.py` — đường dẫn, query embedding (bge + prefix truy vấn), định nghĩa hit, đóng gói context.
- `retrievers.py` — Dense (pgvector), BM25, Hybrid RRF, Rerank (cross-encoder, tùy chọn chấm theo đoạn con), neighbor expansion.
- `eval_retrieval.py` — Hit@1, Evidence Hit@3, Context Evidence Rate, thời gian; trace top-3 + context.
- `agent.py` — Gemini function calling (`search_book`, `final_answer`) trên vòng lặp `starter/agentic.py::run_agent`.
- `run_agent_bench.py` — chạy agent, lưu trace (tool call, query, kết quả, context, câu trả lời, citation, token, thời gian, lỗi).
- `label_agent.py` — chấm tay câu trả lời/citation/abstention/context → `reviews.json`.
