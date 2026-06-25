# 03. Input, Output và vai trò từng khối

## Branch 1: 1D-CNN + BiGRU + Attention

### Input

```text
X_temporal: [T, 132]
```

Trong notebook, dataset trả về tensor theo dạng PyTorch Conv1D:

```text
[B, 132, T]
```

Trong đó:

- `B`: batch size;
- `132`: số feature mỗi frame;
- `T`: số frame theo thời gian.

### Khối 1D-CNN

1D-CNN quét theo trục thời gian để học pattern cục bộ, ví dụ đoạn năng lượng tăng nhanh, giọng chuyển sắc, hoặc spectral contrast thay đổi.

Output khái niệm:

```text
[B, hidden_channels, T']
```

### Khối BiGRU

BiGRU đọc chuỗi theo cả hai chiều thời gian. Nó giúp model nhìn được ngữ cảnh trước và sau mỗi frame.

Output khái niệm:

```text
[B, T', 2*hidden_size]
```

### Khối temporal attention

Attention pooling học frame nào quan trọng hơn cho cảm xúc. Ví dụ trong một câu, đoạn nhấn mạnh hoặc đoạn giọng run có thể quan trọng hơn đoạn im lặng.

Output:

```text
embedding_temporal: [B, 2*hidden_size]
prob_temporal: [B, 6]
```

## Branch 2: Pretrained Spectrogram Encoder / Fallback 2D-CNN + SE

### Input

Nhánh này nhận log-Mel spectrogram từ audio. Nếu dùng pretrained encoder, notebook dùng feature extractor của checkpoint để biến waveform thành input đúng chuẩn encoder.

Dạng khái niệm:

```text
X_spectrogram: [time, mel_bins]
```

Nếu fallback local 2D-CNN:

```text
X_local: [3, mel_bins, time]

channel 1 = log-Mel
channel 2 = delta log-Mel
channel 3 = delta-delta log-Mel
```

### Pretrained spectrogram encoder

Encoder đã học trước trên audio lớn, nên nó có khả năng nhận biết pattern phổ như harmonic, transient, formant-like structure và vùng năng lượng theo thời gian-tần số.

Output:

```text
embedding_spectrogram: [B, hidden_dim]
```

### Classifier head

Classifier head nhỏ học ánh xạ từ embedding sang 6 emotion class.

Output:

```text
prob_spectrogram: [B, 6]
```

## Branch 3: StandardScaler + RBF-SVM

### Input

```text
X_stats: [B, D]
```

`D` là số chiều vector thống kê, phụ thuộc bộ feature được trích xuất.

### StandardScaler

SVM nhạy với scale của feature, nên cần chuẩn hóa:

```text
z = (x - mean_train) / std_train
```

Scaler chỉ fit trên train set để tránh data leakage.

### RBF-SVM

RBF-SVM tạo ranh giới phi tuyến trong không gian feature. Nó phù hợp với statistical vector vì vector này có số chiều vừa phải và chứa nhiều feature thủ công.

Output:

```text
prob_statistical: [B, 6]
```

## Final ensemble

Ba xác suất branch:

```text
p1 = prob_temporal
p2 = prob_spectrogram
p3 = prob_statistical
```

Kết hợp:

```text
p_final = weighted_average(p1, p2, p3)
y_pred = argmax(p_final)
```

Trong đó weight được lấy từ validation macro-F1 của từng nhánh.
