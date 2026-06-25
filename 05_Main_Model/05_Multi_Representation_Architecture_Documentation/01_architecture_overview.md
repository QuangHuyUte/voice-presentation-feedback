# 01. Tổng quan kiến trúc

## Bài toán

Hệ thống nhận đầu vào là một đoạn speech đã chuẩn hóa và dự đoán một trong 6 nhãn cảm xúc chung:

`neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`.

Đây là giai đoạn **emotion recognition** trong roadmap lớn của đề tài. Ở giai đoạn sau, đầu ra emotion có thể được dùng cho **presentation feedback system**, ví dụ phát hiện đoạn nói căng thẳng, thiếu năng lượng, đơn điệu, hoặc thay đổi cảm xúc bất thường theo timeline bài thuyết trình.

## Kiến trúc tổng thể

Kiến trúc hiện tại là **Multi-Representation Feature Engineering Ensemble**. Điểm chính là cùng một audio được trích xuất thành 3 dạng biểu diễn khác nhau, sau đó mỗi biểu diễn được đưa vào một model phù hợp.

```text
Audio 16 kHz
   |
   v
Unified Feature Engineering
   |
   |-- Branch 1: Temporal acoustic sequence
   |       -> 1D-CNN + BiGRU + Attention
   |
   |-- Branch 2: Log-Mel spectrogram
   |       -> Pretrained spectrogram encoder / fallback 2D-CNN + SE
   |
   |-- Branch 3: Handcrafted statistical vector
           -> StandardScaler + RBF-SVM

Final output = validation-weighted ensemble probabilities
```

## Vì sao không dùng gated fusion như Tri-view cũ?

Tri-view gated fusion ép 3 nhánh học chung trong một model. Kết quả thử nghiệm cho thấy nhánh statistical feature có thể giúp ở một vài trường hợp, nhưng dễ làm kém ổn định ở `combined_strict_no_tess` và đặc biệt ở single-dataset nhỏ như SAVEE.

Vì vậy kiến trúc mới chuyển sang **late ensemble**:

- mỗi nhánh học độc lập;
- mỗi nhánh xuất xác suất riêng;
- nhánh nào validation macro-F1 thấp thì weight thấp;
- nhánh yếu không kéo hỏng embedding của nhánh mạnh.

## Ba nhánh chính

### Branch 1: Temporal Acoustic Branch

Nhánh này học sự thay đổi của giọng nói theo thời gian. Cảm xúc không chỉ nằm ở một frame đơn lẻ mà nằm trong cách âm lượng, âm sắc, độ gắt và phổ âm thay đổi xuyên suốt câu nói.

Input:

```text
[T, 132]
```

Output:

```text
p_temporal: xác suất 6 emotion
```

### Branch 2: Pretrained Spectrogram Branch

Nhánh này dùng log-Mel spectrogram như biểu diễn 2D thời gian-tần số. Thay vì train CNN từ đầu trên dataset nhỏ, notebook ưu tiên dùng encoder spectrogram đã pretrained trên audio lớn. Nếu môi trường không tải được checkpoint, nhánh này fallback về 2D-CNN + SE attention.

Input:

```text
waveform audio hoặc log-Mel spectrogram
```

Output:

```text
p_spectrogram: xác suất 6 emotion
```

### Branch 3: Statistical Acoustic Branch

Nhánh này dùng feature engineering thủ công dạng vector cố định. Nó tóm tắt toàn bộ clip bằng các thống kê như mean, std, max, min của MFCC, chroma, spectral và energy features.

Input:

```text
[D] handcrafted statistical vector
```

Output:

```text
p_statistical: xác suất 6 emotion
```

## Công thức ensemble

```text
final_prob = (w1*p_temporal + w2*p_spectrogram + w3*p_statistical) / (w1 + w2 + w3)
```

Trong đó `w1`, `w2`, `w3` được lấy từ validation macro-F1 của từng nhánh. Model cuối chọn class có xác suất cao nhất trong `final_prob`.
