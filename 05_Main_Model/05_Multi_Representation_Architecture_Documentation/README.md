# Multi-Representation Feature Engineering Ensemble

Folder này mô tả chi tiết kiến trúc mô hình chính hiện tại của đồ án Speech Emotion Recognition. Nội dung được viết để có thể đưa trực tiếp vào báo cáo giữa kỳ, slide, hoặc phần giải thích mô hình trong notebook.

Tên kiến trúc đề xuất:

**Multi-Representation Feature Engineering Ensemble for Speech Emotion Recognition**

Ý tưởng cốt lõi:

```text
Input audio 16 kHz
-> Unified feature engineering
-> 3 nhánh chuyên biệt
-> validation-weighted ensemble
-> 6 emotion classes
```

Các file trong folder:

| File | Nội dung |
|---|---|
| `01_architecture_overview.md` | Mô tả kiến trúc tổng thể, vai trò từng nhánh và lý do dùng ensemble |
| `02_feature_engineering.md` | Giải thích từng đặc trưng: lấy từ đâu, dùng ở đâu, vì sao cần |
| `03_branch_input_output.md` | Input/output từng model và các khối xử lý trong từng nhánh |
| `04_training_and_evaluation.md` | Cách train, validation, ensemble và 3 protocol đánh giá |
| `05_report_ready_summary.md` | Đoạn mô tả có thể đưa vào báo cáo |
| `visual_architecture.html` | Bản sơ đồ HTML/CSS có thể mở bằng trình duyệt |
| `diagrams/` | Hình minh họa SVG dùng cho báo cáo/slide |

Các sơ đồ chính:

- `diagrams/architecture_overview.svg`
- `diagrams/feature_engineering_map.svg`
- `diagrams/branch_io_details.svg`
- `diagrams/ensemble_logic.svg`
