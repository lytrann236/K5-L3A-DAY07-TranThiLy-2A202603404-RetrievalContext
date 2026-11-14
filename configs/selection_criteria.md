# Tiêu chí chọn cấu hình (ghi ngày 2026-11-14, sau khi xem retrieval dev, TRƯỚC khi chạy agent dev đầy đủ và trước mọi lần chạy test)

Ứng viên: M1_dense (baseline), I1_hybrid_rerank, I3_rerank_maxseg. I2_neighbors bị loại: không tăng chỉ số nào nhưng context gấp đôi.

Thứ tự ưu tiên trên dev:
1. Context Evidence Rate (retrieval) — agent chỉ trả lời đúng khi evidence có trong context.
2. Hành vi agent trên dev: số câu có đáp án bị từ chối nhầm (thấp hơn tốt hơn) và số câu thiếu bằng chứng được từ chối (cao hơn tốt hơn).
3. Hit@1 (retrieval).
4. Thời gian/câu và số request Gemini (thấp hơn tốt hơn). Chi phí đều 0 USD.

Test chỉ chạy một lần cho baseline (M1_dense) và cấu hình được chọn, sau khi ghi configs/selected.json.
