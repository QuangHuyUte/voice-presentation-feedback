# 04. Training và Evaluation

## Quy trình train

Với mỗi experiment protocol, notebook train/evaluate các nhánh độc lập.

```text
1. Load metadata và audio/features
2. Split train/validation/test theo protocol
3. Fit Branch 1 trên temporal features
4. Fit Branch 2 trên spectrogram/pretrained embeddings
5. Fit Branch 3 trên statistical vector
6. Tính validation macro-F1 từng nhánh
7. Ensemble theo validation macro-F1
8. Báo cáo accuracy, macro-F1, confusion matrix và per-dataset metrics
```

## Model selection

Mỗi deep branch dùng validation macro-F1 để chọn checkpoint tốt nhất. Macro-F1 quan trọng vì dataset có thể mất cân bằng giữa emotion hoặc giữa corpus. Nếu chỉ nhìn accuracy, model có thể đoán tốt class đông nhưng bỏ qua class ít.

## Weighted ensemble

Ví dụ:

```text
Branch 1 validation macro-F1 = 0.72
Branch 2 validation macro-F1 = 0.69
Branch 3 validation macro-F1 = 0.61

weights = [0.72, 0.69, 0.61]
```

Khi đó Branch 1 đóng góp lớn nhất, nhưng Branch 2 và Branch 3 vẫn được dùng nếu chúng có tín hiệu bổ sung.

## Ba protocol đánh giá

### 1. combined_random

Gộp tất cả dataset rồi chia random train/validation/test. Protocol này gần với nhiều paper điểm cao, nhưng có rủi ro cùng speaker hoặc cùng corpus pattern xuất hiện ở train và test. Vì vậy kết quả thường cao hơn strict split.

### 2. combined_strict_no_tess

Protocol khó hơn. Dùng split speaker-aware hoặc corpus-aware, loại TESS khỏi strict vì TESS có rất ít speaker và dễ làm split không ổn định. Protocol này kiểm tra generalization tốt hơn: model có nhận diện cảm xúc trên speaker/corpus chưa thấy hay không.

### 3. single-dataset experiments

Train/test riêng trên từng dataset:

```text
single_RAVDESS
single_CREMA-D
single_TESS
single_SAVEE
```

Mục tiêu là so sánh với các paper thường báo kết quả theo từng dataset.

## Output cần có

| File output | Ý nghĩa |
|---|---|
| `multi_rep_ensemble_metrics.csv` | Kết quả từng branch và ensemble |
| `multi_rep_ensemble_summary.json` | Tóm tắt cấu hình chạy |
| `classification_report_*.csv` | Precision/recall/F1 theo class |
| `confusion_matrix_*.png` | Ma trận nhầm lẫn |
| `per_dataset_metrics.csv` | Kết quả theo từng dataset nếu test gộp |
