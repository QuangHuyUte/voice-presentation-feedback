# 05. Mô tả đưa vào báo cáo

Mô hình đề xuất trong đồ án là **Multi-Representation Feature Engineering Ensemble** cho bài toán Speech Emotion Recognition. Thay vì chỉ sử dụng một loại đặc trưng, hệ thống trích xuất ba dạng biểu diễn từ cùng một audio đã chuẩn hóa 16 kHz: biểu diễn chuỗi thời gian, biểu diễn phổ 2D, và vector thống kê âm học. Ba biểu diễn này được đưa vào ba nhánh mô hình chuyên biệt rồi kết hợp bằng validation-weighted ensemble.

Nhánh thứ nhất là **Temporal Acoustic Branch**. Nhánh này sử dụng các đặc trưng theo frame gồm MFCC, delta MFCC, delta-delta MFCC, RMS, ZCR, spectral centroid, spectral bandwidth, spectral rolloff và spectral contrast. Các đặc trưng này tạo thành chuỗi thời gian có 132 chiều mỗi frame. Chuỗi này đi qua 1D-CNN để học pattern cục bộ, BiGRU để học ngữ cảnh hai chiều theo thời gian, và temporal attention pooling để chọn các frame quan trọng nhất cho cảm xúc.

Nhánh thứ hai là **Pretrained Spectrogram Branch**. Nhánh này sử dụng log-Mel spectrogram, tức biểu diễn 2D thời gian-tần số của tín hiệu speech. Notebook ưu tiên dùng một audio-pretrained spectrogram encoder để tận dụng tri thức đã học từ dữ liệu audio lớn. Sau encoder, một classifier head nhỏ được train để dự đoán 6 emotion classes. Nếu môi trường không tải được pretrained checkpoint, notebook tự fallback sang 2D-CNN + SE attention trên ảnh log-Mel 3 kênh.

Nhánh thứ ba là **Statistical Acoustic Branch**. Nhánh này dùng vector thống kê toàn clip gồm các thống kê mean, std, min, max của MFCC, delta, delta-delta, chroma, spectral features, RMS, ZCR, energy và entropy of energy. Vector này được chuẩn hóa bằng StandardScaler và phân loại bằng RBF-SVM. Đây là nhánh đại diện cho feature engineering thủ công, giúp bổ sung các thông tin âm học toàn cục mà deep branch có thể chưa học ổn định trên dataset nhỏ.

Ba nhánh lần lượt tạo ra xác suất dự đoán `p1`, `p2`, `p3` cho 6 cảm xúc: neutral, happy, sad, angry, fear và disgust. Kết quả cuối cùng được tính bằng weighted ensemble:

```text
final_prob = (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3)
```

Trong đó `w1`, `w2`, `w3` được tính từ validation macro-F1 của từng nhánh. Cách này giúp hệ thống tin hơn vào nhánh có validation performance tốt, đồng thời giảm ảnh hưởng của nhánh yếu. So với gated fusion trước đó, ensemble ổn định hơn vì mỗi nhánh được train độc lập và không ép feature thủ công trộn trực tiếp vào embedding của deep model.
