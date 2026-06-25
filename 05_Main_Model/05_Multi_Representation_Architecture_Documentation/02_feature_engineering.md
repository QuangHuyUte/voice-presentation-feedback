# 02. Feature Engineering

Feature engineering trong đồ án này không chỉ có nghĩa là tạo một vector thủ công. Nó là tầng thống nhất biến audio thô thành nhiều biểu diễn phù hợp cho từng model.

## Chuẩn hóa audio

Mỗi file audio được chuẩn hóa:

```text
sample rate: 16 kHz
channel: mono
duration: cố định theo cấu hình notebook
amplitude: normalized
```

Mục tiêu là giảm khác biệt kỹ thuật giữa RAVDESS, CREMA-D, TESS và SAVEE trước khi trích xuất đặc trưng.

## Nhóm A: Temporal sequence features

Dạng dữ liệu:

```text
1 audio -> T frames -> 132 features/frame
```

Các đặc trưng:

| Feature | Số chiều/frame | Lấy từ đâu | Vì sao dùng |
|---|---:|---|---|
| MFCC | 40 | `librosa.feature.mfcc` | Mô tả âm sắc và spectral envelope của giọng nói |
| Delta MFCC | 40 | `librosa.feature.delta(mfcc)` | Mô tả tốc độ thay đổi âm sắc |
| Delta-delta MFCC | 40 | `librosa.feature.delta(mfcc, order=2)` | Mô tả gia tốc thay đổi, giúp bắt pattern động |
| RMS | 1 | `librosa.feature.rms` | Cường độ/năng lượng giọng nói |
| ZCR | 1 | `librosa.feature.zero_crossing_rate` | Độ gắt, texture voiced/unvoiced |
| Spectral centroid | 1 | `librosa.feature.spectral_centroid` | Độ sáng của phổ âm |
| Spectral bandwidth | 1 | `librosa.feature.spectral_bandwidth` | Độ rộng phổ |
| Spectral rolloff | 1 | `librosa.feature.spectral_rolloff` | Mức tập trung năng lượng ở vùng tần số cao |
| Spectral contrast | 7 | `librosa.feature.spectral_contrast` | Độ tương phản giữa đỉnh và đáy phổ |

Lý do đưa nhóm này vào Branch 1: các feature này giữ trục thời gian, phù hợp với 1D-CNN, GRU/LSTM và attention.

## Nhóm B: Log-Mel spectrogram

Dạng dữ liệu gốc:

```text
1 audio -> log-Mel spectrogram
```

Log-Mel spectrogram có thể hiểu như một ảnh của âm thanh:

- trục ngang: thời gian;
- trục dọc: tần số theo thang Mel;
- pixel/value: năng lượng âm thanh.

Nhánh này dùng encoder spectrogram pretrained trên audio lớn. Trong notebook, checkpoint mặc định là:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

Điểm quan trọng: kiến trúc tổng thể không đặt tên là AST, mà mô tả là **pretrained spectrogram branch**. AST chỉ là checkpoint/backbone mặc định và có thể thay đổi sau.

Nếu checkpoint không tải được, fallback local dùng:

```text
channel 1: log-Mel
channel 2: delta log-Mel
channel 3: delta-delta log-Mel
```

Lý do đưa nhóm này vào Branch 2: spectrogram giữ cấu trúc thời gian-tần số, phù hợp với pretrained audio/spectrogram encoder hoặc 2D-CNN.

## Nhóm C: Statistical vector features

Dạng dữ liệu:

```text
1 audio -> 1 vector cố định
```

Các thống kê được lấy trên toàn clip:

| Feature group | Statistics | Ý nghĩa |
|---|---|---|
| MFCC | mean/std/min/max | Âm sắc tổng thể và độ biến thiên |
| Delta MFCC | mean/std | Mức thay đổi âm sắc |
| Delta-delta MFCC | mean/std | Gia tốc thay đổi âm sắc |
| Chroma STFT/CQT/CENS | mean/std | Harmonic và pitch-class structure |
| Spectral centroid | mean/std/max | Độ sáng phổ |
| Spectral bandwidth | mean/std/max | Độ rộng phổ |
| Spectral rolloff | mean/std/max | Vùng tần số chứa phần lớn năng lượng |
| Spectral contrast | mean/std | Tương phản phổ |
| RMS | mean/std/max | Cường độ nói |
| ZCR | mean/std/max | Độ gắt/voiced-unvoiced texture |
| Energy | mean/std/max | Năng lượng toàn clip |
| Entropy of energy | mean/std/max | Năng lượng tập trung hay phân tán |

Deep model học pattern từ dữ liệu, còn statistical branch cung cấp tri thức âm học rõ ràng. Nó giống một bản tóm tắt toàn clip giúp hệ thống có thêm góc nhìn khác, đặc biệt hữu ích khi kết hợp ensemble.
