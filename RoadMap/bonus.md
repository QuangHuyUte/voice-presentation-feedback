# Đề xuất nâng cấp notebook SER theo hướng Domain-Robust + Speaker-Invariant

**Bối cảnh dự án.** Bạn đang xây dựng hệ thống **Speech Emotion Recognition (SER)** trên 4 dataset emotion. Hiện có 2 protocol chính: (1) gộp 4 dataset rồi chia đều/random, và (2) strict split sao cho speaker trong train/validation/test khác nhau. Kết quả hiện tại khoảng **80% với random split** và **60% với strict speaker split**. Mục tiêu xa hơn là **realtime presentation feedback**: phát hiện dấu hiệu run, thiếu tự tin, thay đổi nhịp nói, năng lượng giọng và đưa ra phản hồi cho người thuyết trình.

**Kết luận chính.** Không nên chạy theo mốc 90% random split của một số paper, vì random split có thể bị speaker leakage hoặc domain leakage. Với bài toán của bạn, điểm nghiên cứu mạnh hơn là: mô hình vẫn giữ được hiệu năng tốt khi gặp **speaker mới**, **dataset mới**, hoặc **môi trường thu âm mới**.

---

## 1. Notebook hiện tại đang có gì?

Qua notebook `06_Advanced_Multi_Representation_Domain_Robust_SER.ipynb`, kiến trúc hiện tại đã khá mạnh:

```text
Audio 16 kHz
  |-- Branch A: temporal acoustic features
  |      MFCC + delta + delta-delta + RMS + ZCR + spectral features
  |      -> 1D-CNN -> BiLSTM -> attention pooling -> z_temporal
  |
  |-- Branch B: spectrogram branch
  |      log-Mel + delta + delta-delta
  |      -> 2D-CNN / residual blocks + SE attention -> z_spectral
  |
  |-- Branch C: pretrained speech branch
  |      raw waveform -> frozen WavLM-base-plus -> adapter MLP -> z_wavlm
  |
  |-- Branch D: statistical branch
         handcrafted statistical vector -> stats MLP -> z_stats

Fusion:
  z_fused = concat(z_temporal, z_spectral, z_wavlm, z_stats)
      |-- emotion head
      |-- optional domain adversarial head + GRL

Final:
  deep fusion probability + RBF-SVM probability -> stacking ensemble
```

Notebook cũng đã có:

- `combined_random`: gần với cách nhiều repo/paper báo cáo random split.
- `combined_strict_no_tess`: strict hơn theo speaker, loại TESS khỏi strict vì TESS chỉ có rất ít speaker.
- `single-dataset`: train/test riêng từng corpus.
- WavLM frozen embedding.
- Domain-adversarial head với Gradient Reversal Layer (GRL).
- Mixup, SpecAugment, waveform augmentation.
- Class weights, label smoothing, confusion matrix, macro-F1.

**Điểm thiếu quan trọng nhất:** hiện tại notebook mới có domain adversarial theo dataset, nhưng vấn đề bạn đang gặp mạnh nhất là strict speaker split giảm nhiều. Vì vậy, nâng cấp nên ưu tiên **speaker-adversarial head** trước khi thử các hướng quá xa như voice conversion hoặc test-time adaptation đầy đủ.

---

## 2. Vấn đề khoa học cần giải quyết

### 2.1 Vì sao random split cao nhưng strict split thấp?

Trong SER, mỗi sample không chỉ chứa cảm xúc. Nó còn chứa nhiều tín hiệu phụ:

```text
x_audio = emotion signal + speaker identity + microphone + room noise + acting style + language/accent + dataset artifact
```

Nếu random split cho phép cùng speaker xuất hiện ở train và test, mô hình có thể học:

```text
speaker A thường nói angry
speaker B có pitch cao nên hay bị đoán happy
TESS sạch và diễn rõ nên dễ đoán
CREMA-D có môi trường khác nên embedding tách ra riêng
```

Khi chuyển sang strict split, speaker test không có trong train, các shortcut này mất tác dụng, accuracy giảm. Vì vậy, strict split 60% không nhất thiết là thất bại; nó cho thấy bạn đang đo đúng bài toán generalization.

### 2.2 Mục tiêu mới của mô hình

Mục tiêu không chỉ là:

```text
maximize accuracy trên random split
```

mà là:

```text
Học embedding z sao cho:
- z vẫn chứa đủ thông tin emotion.
- z bớt chứa thông tin dataset/domain.
- z bớt chứa thông tin speaker identity.
- z ổn định hơn khi gặp speaker/domain mới.
```

---

## 3. Kiến trúc đề xuất sau khi nâng cấp

```text
Audio
  -> WavLM / HuBERT / wav2vec2 hoặc embedding WavLM frozen hiện tại
  -> projection / adapter
  -> fused embedding z
       |-- emotion classifier
       |-- domain adversarial head + GRL
       |-- speaker adversarial head + GRL
       |-- optional supervised contrastive projection head
```

Trong notebook hiện tại, phần hợp lý nhất là **giữ multi-branch fusion đang có**, sau đó thêm nhánh speaker và contrastive vào `z_fused`:

```text
z_fused = concat(z_temporal, z_spectral, z_wavlm, z_stats)

z_fused
  |-- Emotion head: predict emotion class
  |-- Domain head + GRL: predict dataset/domain, but force encoder to hide domain cues
  |-- Speaker head + GRL: predict speaker_id, but force encoder to hide speaker cues
  |-- SupCon head: make same-emotion samples closer in embedding space
```

---

## 4. Các thuật ngữ chính

### 4.1 WavLM / HuBERT / wav2vec2

Đây là các **self-supervised pretrained speech encoders**. Chúng được pretrain trên rất nhiều audio không nhãn, sau đó dùng làm backbone/feature extractor cho task downstream như SER.

- **wav2vec 2.0** mask latent speech representations rồi học contrastive task để đoán phần bị che. Nó chứng minh speech representation có thể học hiệu quả từ audio không nhãn trước khi fine-tune. Nguồn: Baevski et al., 2020. Paper: https://arxiv.org/abs/2006.11477
- **HuBERT** dùng offline clustering để tạo hidden-unit targets, sau đó học masked prediction trên vùng bị che. Nguồn: Hsu et al., 2021. Paper/PDF: https://arxiv.org/abs/2106.07447 / https://arxiv.org/pdf/2106.07447
- **WavLM** được thiết kế cho full-stack speech processing, không chỉ ASR. WavLM học masked speech prediction kết hợp denoising, phù hợp hơn với nhiều task speech như speaker, paralinguistic và emotion. Nguồn: Chen et al., 2021. Paper/PDF: https://arxiv.org/abs/2110.13900 / https://arxiv.org/pdf/2110.13900. Repo chính thức: https://github.com/microsoft/unilm/tree/master/wavlm

Với notebook hiện tại, bạn đang dùng `microsoft/wavlm-base-plus` theo hướng hợp lý: **frozen WavLM + adapter**, giúp nhẹ GPU và giảm overfit.

### 4.2 Projection / Adapter

WavLM thường xuất embedding lớn, ví dụ 768 chiều. Adapter là một MLP nhỏ để chuyển embedding lớn thành embedding phù hợp hơn cho emotion:

```text
h_wavlm in R^768
z_wavlm = Adapter(h_wavlm) in R^d
```

Ví dụ:

```text
Linear(768 -> 256) -> GELU -> Dropout -> Linear(256 -> 192)
```

Lý do cần adapter:

- WavLM biết nhiều thứ: nội dung lời nói, speaker, noise, prosody.
- SER chỉ cần phần liên quan đến cảm xúc.
- Adapter giúp lọc và tái ánh xạ representation cho task emotion.

### 4.3 Emotion classifier

Emotion head là nhánh chính, dùng để dự đoán class cảm xúc:

```text
z -> emotion head -> logits -> softmax probability
```

Công thức:

```text
p_y = softmax(W_e z + b_e)
```

Với one-hot label y, Cross Entropy loss:

```text
L_emo = - sum_{c=1}^{C} y_c log(p_{y,c})
```

Trong notebook có thể dùng `CrossEntropyLoss`, class weights và label smoothing.

### 4.4 Domain adversarial head

Domain là nguồn dữ liệu/dataset:

```text
RAVDESS -> domain 0
CREMA-D -> domain 1
SAVEE   -> domain 2
TESS    -> domain 3
```

Domain head cố đoán dataset từ embedding:

```text
p_d = softmax(W_d GRL(z) + b_d)
L_domain = - sum_{k=1}^{K} d_k log(p_{d,k})
```

Nhưng vì đi qua GRL, encoder bị cập nhật theo hướng làm domain head khó đoán domain hơn. Mục tiêu là giảm dataset leakage.

Nguồn nền tảng: Ganin et al., **Domain-Adversarial Training of Neural Networks**. Paper: https://arxiv.org/abs/1505.07818. PDF JMLR: https://www.jmlr.org/papers/volume17/15-239/15-239.pdf. Reference PyTorch implementation không chính thức: https://github.com/fungtion/DANN, https://github.com/NaJaeMin92/pytorch-DANN

Nguồn áp dụng trực tiếp cho acoustic emotion recognition: Abdelwahab & Busso, **Domain Adversarial for Acoustic Emotion Recognition**. Paper/PDF: https://arxiv.org/abs/1804.07690 / https://lab-msp.com/MSP/publications/Abdelwahab_2018_2.pdf

### 4.5 Speaker adversarial head

Speaker head cố đoán `speaker_id` từ embedding:

```text
p_s = softmax(W_s GRL(z) + b_s)
L_speaker = - sum_{m=1}^{M} s_m log(p_{s,m})
```

Nhưng GRL đảo gradient, nên encoder bị ép làm embedding khó nhận diện speaker hơn.

Nguồn trực tiếp: Tu et al., **Towards adversarial learning of speaker-invariant representation for speech emotion recognition**. Paper/PDF: https://arxiv.org/abs/1903.09606 / https://arxiv.org/pdf/1903.09606. Paper đề xuất representation network + emotion classifier + speaker classifier, dùng adversarial training để học speaker-invariant representation.

Nguồn bổ sung: Li et al., **Speaker-invariant Affective Representation Learning via Adversarial Training**. Paper: https://arxiv.org/abs/1911.01533. Repo liên quan speaker-invariant DANN cho emotion: https://github.com/ihp-lab/Speaker-Invariant-Domain-Adversarial-Neural-Networks

### 4.6 Gradient Reversal Layer (GRL)

GRL không đổi giá trị ở forward:

```text
GRL(z) = z
```

Nhưng ở backward, GRL đảo dấu gradient:

```text
d GRL(z) / dz = -lambda I
```

Do đó:

- domain/speaker head học để đoán đúng domain/speaker;
- encoder/fusion học theo hướng ngược lại, làm domain/speaker head khó đoán hơn.

Cách hiểu min-max:

```text
min_{theta_f, theta_e} L_emo(theta_f, theta_e) - lambda_d L_domain(theta_f, theta_d)
min_{theta_d} L_domain(theta_f, theta_d)
```

Với speaker:

```text
min_{theta_f, theta_e} L_emo(theta_f, theta_e) - lambda_s L_speaker(theta_f, theta_s)
min_{theta_s} L_speaker(theta_f, theta_s)
```

Trong code thực tế với GRL, ta thường viết tổng loss bình thường:

```text
L_total = L_emo + beta_d L_domain + beta_s L_speaker
```

Nhưng do gradient đi qua GRL bị nhân `-lambda`, encoder nhận gradient adversarial.

---

## 5. Những nâng cấp nên áp dụng ngay cho notebook hiện tại

### 5.1 Nâng cấp 1 - Thêm speaker-adversarial head

**Mức ưu tiên:** rất cao.  
**Lý do:** strict speaker split đang là điểm yếu chính. Domain adversarial chỉ giảm dataset leakage, chưa trực tiếp xử lý speaker leakage.

#### Vấn đề

Nếu embedding còn chứa nhiều thông tin speaker, mô hình có thể học shortcut:

```text
speaker identity -> emotion label
```

Điều này làm random split cao nhưng strict split giảm.

#### Cách thêm vào notebook

Hiện metadata đã có `speaker_id`. Thêm speaker code:

```python
metadata["speaker_code"] = metadata["speaker_id"].astype("category").cat.codes.astype(int)
speaker_y = metadata["speaker_code"].to_numpy().astype(np.int64)
NUM_SPEAKERS = int(metadata["speaker_code"].nunique())
```

Trong `MultiBranchDataset`, trả thêm speaker label:

```python
return {
    "temporal": temporal_tensor,
    "spectral": spectral_tensor,
    "stats": stats_tensor,
    "speech": speech_tensor,
    "label": torch.tensor(y[i], dtype=torch.long),
    "domain": torch.tensor(domain_y[i], dtype=torch.long),
    "speaker": torch.tensor(speaker_y[i], dtype=torch.long),
}
```

Trong model, thêm speaker head:

```python
self.speaker_head = nn.Sequential(
    nn.Linear(fusion_dim, 128),
    nn.GELU(),
    nn.Dropout(DROPOUT),
    nn.Linear(128, num_speakers)
)
```

Trong forward:

```python
emo_logits = self.emotion_head(z)
dom_logits = self.domain_head(grad_reverse(z, grl_lambda))
spk_logits = self.speaker_head(grad_reverse(z, grl_lambda))
return emo_logits, dom_logits, spk_logits, z
```

Loss:

```text
L_total = L_emo + beta_d L_domain + beta_s L_speaker
```

Khuyến nghị ban đầu:

```text
ADV_LAMBDA_MAX = 0.03 hoặc 0.05
DOMAIN_LOSS_WEIGHT = 0.5
SPEAKER_LOSS_WEIGHT = 0.2 đến 0.3
ADV_WARMUP_EPOCHS = 5
```

#### Vì sao không set speaker adversarial quá mạnh?

Trong speech, speaker cue và emotion cue chồng nhau:

- pitch vừa là đặc trưng speaker/giới tính, vừa liên quan đến emotion;
- energy vừa là thói quen nói, vừa là tín hiệu angry/excited;
- timbre vừa là identity, vừa thay đổi theo cảm xúc.

Nếu ép embedding mất speaker quá mạnh, model có thể mất luôn thông tin emotion. Vì vậy phải dùng warm-up và lambda nhỏ.

#### Dẫn chứng

- Tu et al., 2019: adversarial training cho speaker-invariant SER, gồm representation network, emotion classifier và speaker classifier. Paper báo cáo cải thiện so với baseline trong speaker-invariant setting. Link: https://arxiv.org/abs/1903.09606
- Li et al., 2019: dùng GRL và entropy loss để giảm speaker information trong affective representation. Link: https://arxiv.org/abs/1911.01533
- Repo liên quan: https://github.com/ihp-lab/Speaker-Invariant-Domain-Adversarial-Neural-Networks

---

### 5.2 Nâng cấp 2 - Làm ablation cho domain adversarial/DANN

**Mức ưu tiên:** rất cao.  
**Lý do:** notebook đã có domain adversarial, nhưng cần chứng minh nó thật sự giúp generalization hoặc ít nhất giảm domain leakage.

#### Vấn đề

Khi gộp 4 dataset, model có thể học dataset artifact:

```text
TESS: sạch, diễn rõ, ít speaker
SAVEE: số speaker ít, đặc trưng nam
RAVDESS: acted speech, format cố định
CREMA-D: nhiều speaker hơn, chất lượng và acting style khác
```

Model có thể học:

```text
dataset style -> emotion
```

thay vì học emotion thật.

#### Công thức DANN

Feature extractor tạo embedding:

```text
z = F(x; theta_f)
```

Emotion classifier:

```text
p_y = C_y(z; theta_y)
L_emo = CE(y, p_y)
```

Domain classifier:

```text
p_d = C_d(GRL(z); theta_d)
L_domain = CE(d, p_d)
```

Tổng loss khi train bằng GRL:

```text
L_total = L_emo + beta_d L_domain
```

Nhưng gradient về feature extractor từ `L_domain` bị đảo dấu:

```text
grad_z_from_domain = -lambda * dL_domain/dz
```

#### Ablation nên chạy

```text
A0: no adversarial
A1: domain adversarial only, lambda=0.02
A2: domain adversarial only, lambda=0.05
A3: speaker adversarial only, lambda=0.03
A4: domain + speaker adversarial
A5: domain + speaker adversarial + supervised contrastive
```

Metric cần báo cáo:

```text
accuracy
macro-F1
UAR / balanced accuracy
domain accuracy
speaker accuracy
per-dataset macro-F1
confusion matrix
```

Cách đọc kết quả:

- Nếu `domain_accuracy` cao: embedding vẫn chứa dấu vết dataset.
- Nếu bật DANN làm random split giảm nhẹ nhưng strict/cross-domain tăng: đó là kết quả tốt cho mục tiêu domain-robust.
- Nếu emotion macro-F1 giảm mạnh: lambda quá lớn hoặc warm-up quá ngắn.

#### Dẫn chứng

- Ganin et al., 2016: DANN dùng GRL để học feature discriminative cho task chính nhưng khó phân biệt domain. Link: https://arxiv.org/abs/1505.07818, PDF: https://www.jmlr.org/papers/volume17/15-239/15-239.pdf
- Abdelwahab & Busso, 2018: áp dụng adversarial multitask learning cho acoustic emotion recognition để học representation chung giữa train/test domain. Link: https://arxiv.org/abs/1804.07690, PDF: https://lab-msp.com/MSP/publications/Abdelwahab_2018_2.pdf
- Self-supervised adversarial domain adaptation cho cross-corpus/cross-language SER: https://arxiv.org/abs/2204.08625, PDF: https://opus.bibliothek.uni-augsburg.de/opus4/files/108969/108969.pdf

---

### 5.3 Nâng cấp 3 - Thêm supervised contrastive learning nhẹ

**Mức ưu tiên:** cao, nhưng nên làm sau speaker-adversarial head.  
**Lý do:** CrossEntropy chỉ học ranh giới phân loại. SupCon ép embedding của các mẫu cùng emotion gần nhau và khác emotion xa nhau. Điều này phù hợp khi bạn muốn embedding ổn định hơn qua speaker/domain.

#### Vấn đề

Với strict split, cùng một emotion có thể biểu hiện khác nhau theo speaker:

```text
angry của speaker nam trầm != angry của speaker nữ cao
sad của TESS != sad của CREMA-D
neutral của RAVDESS != neutral của SAVEE
```

Nếu chỉ dùng CrossEntropy, model có thể học decision boundary dựa vào shortcut. SupCon giúp học cấu trúc embedding:

```text
same emotion -> closer
opposite/different emotion -> farther
```

#### Công thức SupCon

Với batch embeddings `z_i`, normalize thành `v_i`:

```text
v_i = z_i / ||z_i||
```

Tập positive của sample i:

```text
P(i) = {p != i | y_p = y_i}
```

Supervised contrastive loss:

```text
L_supcon = sum_i [ - 1/|P(i)| * sum_{p in P(i)} log exp(v_i · v_p / tau) / sum_{a != i} exp(v_i · v_a / tau) ]
```

Với bài của bạn nên ưu tiên positive pair:

```text
same emotion + different speaker
same emotion + different dataset
```

Tổng loss:

```text
L_total = L_emo + alpha L_supcon + beta_d L_domain + beta_s L_speaker
```

Khuyến nghị ban đầu:

```text
SUPCON_WEIGHT = 0.05 hoặc 0.1
TEMPERATURE tau = 0.1
```

#### Lưu ý thực nghiệm

- SupCon cần batch có đủ nhiều class và đủ positive pairs.
- Nếu batch size nhỏ, loss có thể nhiễu.
- Nếu class imbalance mạnh, nên dùng weighted sampler hoặc batch sampler cân bằng emotion.

#### Dẫn chứng

- Khosla et al., **Supervised Contrastive Learning**, 2020. Paper: https://arxiv.org/abs/2004.11362. Reference repo: https://github.com/HobbitLong/SupContrast
- Xiang, **A Cross-Corpus Speech Emotion Recognition Method Based on Supervised Contrastive Learning**, 2024. Paper/PDF: https://arxiv.org/abs/2411.19803 / https://arxiv.org/pdf/2411.19803. Paper này dùng WavLM-based model và two-stage fine-tuning với supervised contrastive learning cho cross-corpus SER.
- ECAN source-free cross-corpus SER cũng dùng nearest-neighbor contrastive và supervised contrastive để tăng emotion consistency. Link: https://arxiv.org/abs/2401.12925

---

### 5.4 Nâng cấp 4 - Thêm leave-one-dataset-out evaluation

**Mức ưu tiên:** cao.  
**Lý do:** Nếu đề tài nói domain-robust, phải kiểm tra khi test domain không xuất hiện trong train.

#### Protocol đề xuất

```text
Train: RAVDESS + CREMA-D + SAVEE
Test: TESS

Train: RAVDESS + CREMA-D + TESS
Test: SAVEE

Train: RAVDESS + SAVEE + TESS
Test: CREMA-D

Train: CREMA-D + SAVEE + TESS
Test: RAVDESS
```

Có thể thêm validation bằng cách tách từ train domains.

#### Vì sao cần?

Random split trả lời:

```text
Model có học được task trên phân phối tương tự không?
```

Strict speaker split trả lời:

```text
Model có chịu được speaker mới không?
```

Leave-one-dataset-out trả lời:

```text
Model có chịu được domain/dataset mới không?
```

Với presentation feedback, người dùng thật sẽ là domain mới: micro mới, phòng mới, tiếng Việt hoặc accent mới, phong cách nói tự nhiên hơn acted dataset.

#### Cần báo cáo gì?

```text
Dataset held-out
Accuracy
Macro-F1
UAR
Per-class F1
Confusion matrix
```

Đặc biệt cần ghi chú TESS vì TESS có ít speaker và acted speech rất sạch, nên kết quả có thể không đại diện cho môi trường thực.

#### Dẫn chứng

- Cross-corpus SER thường gặp degradation do train/test khác corpus/distribution. Điều này được nêu rõ trong Domain Adversarial for Acoustic Emotion Recognition: https://arxiv.org/abs/1804.07690
- Self-supervised adversarial domain adaptation cũng đặt vấn đề lack of generalization trong cross-corpus/cross-language SER: https://arxiv.org/abs/2204.08625

---

### 5.5 Nâng cấp 5 - Thêm diagnostics theo domain/speaker/class

**Mức ưu tiên:** rất cao.  
**Lý do:** Accuracy tổng không đủ để debug SER.

Cần thêm bảng:

```text
per-dataset macro-F1
per-emotion precision/recall/F1
per-speaker-group performance
domain accuracy
speaker accuracy
confusion matrix normalized by true label
```

Cách đọc:

```text
Nếu fear/sad hay nhầm -> có thể feature prosody chưa đủ hoặc label mapping không nhất quán.
Nếu CREMA-D thấp hơn hẳn -> domain mismatch mạnh.
Nếu speaker accuracy cao dù có speaker GRL -> embedding vẫn chứa speaker identity.
Nếu domain accuracy gần random chance -> domain-invariant tốt hơn.
```

Ví dụ với 4 domain, domain accuracy random chance khoảng:

```text
1 / 4 = 25%
```

Nếu domain classifier vẫn đạt 80-90%, embedding còn domain leakage mạnh. Nếu giảm về gần 25-40% mà emotion macro-F1 không giảm mạnh, DANN đang hoạt động đúng hướng.

---

### 5.6 Nâng cấp 6 - Realtime smoothing và speaker baseline cho demo

**Mức ưu tiên:** cao cho phần ứng dụng, nhưng không cần thay đổi model lớn.  
**Lý do:** Presentation feedback không nên chỉ in ra `fear = 80%`. Người dùng cần phản hồi hành vi: nói nhanh, năng lượng giảm, nhiều pause, giọng căng hơn baseline.

#### Pipeline demo

```text
Microphone / long audio file
  -> VAD lọc đoạn có tiếng nói
  -> window 2-3 giây, hop 1 giây
  -> model dự đoán emotion probability
  -> smoothing theo thời gian
  -> so sánh với speaker baseline
  -> feedback text
```

#### Smoothing công thức

```text
p_smooth(t) = alpha p_smooth(t-1) + (1 - alpha) p_model(t)
```

Ví dụ:

```text
alpha = 0.7
```

#### Speaker baseline

Trong 30-60 giây đầu hoặc đoạn calibration, lấy baseline:

```text
baseline_pitch_mean
baseline_energy_mean
baseline_speech_rate
baseline_pause_ratio
```

Sau đó feedback theo trend:

```text
Nếu speech_rate > baseline + threshold:
  "Tốc độ nói đang nhanh hơn bình thường, nên giảm nhịp."

Nếu energy giảm liên tục:
  "Năng lượng giọng đang giảm ở cuối đoạn."

Nếu fear/tension probability tăng + pitch variability tăng:
  "Giọng có xu hướng căng hơn, nên ngắt nhịp và hít thở trước câu tiếp theo."
```

#### Vì sao nên làm dạng feedback thay vì emotion label tuyệt đối?

Trong thuyết trình, cảm xúc không luôn là class rời rạc rõ ràng. "Run", "thiếu tự tin", "căng" là trạng thái biểu hiện qua nhiều tín hiệu: pause, tốc độ, energy, pitch stability, voice tremor, articulation. Vì vậy output nên là **trend + behavioral feedback**.

---

## 6. Những hướng nên để future work

### 6.1 Cross-lingual SER

**Để future work**, trừ khi bạn có dataset tiếng Việt hoặc nhiều ngôn ngữ.

Nếu 4 dataset hiện tại chủ yếu là tiếng Anh, không nên gọi là cross-lingual. Nên gọi đúng là:

```text
cross-corpus SER
speaker-independent SER
domain-robust SER
```

Future work có thể viết:

```text
Future work will extend the system to Vietnamese presentation speech and investigate cross-lingual emotion adaptation.
```

Dẫn chứng: Latif et al., **Unsupervised Adversarial Domain Adaptation for Cross-Lingual Speech Emotion Recognition**, PDF: https://arxiv.org/pdf/1907.06083. Self-supervised ADDi/sADDi cũng mở rộng sang cross-corpus/cross-language SER: https://arxiv.org/abs/2204.08625

### 6.2 Voice conversion cho domain adaptation

**Để future work.**

Ý tưởng voice conversion là chuyển giọng/domain của unlabeled target sang gần labeled source domain hơn. Bài Interspeech 2024 dùng k-nearest neighbors voice conversion cho unsupervised domain adaptation trong SER.

Dẫn chứng: Mote et al., **Unsupervised Domain Adaptation for Speech Emotion Recognition using K-Nearest Neighbors Voice Conversion**, Interspeech 2024. PDF: https://www.isca-archive.org/interspeech_2024/mote24_interspeech.pdf

Vì sao chưa nên làm ngay:

- Cần pipeline voice conversion riêng.
- Có nguy cơ làm mất cue cảm xúc như pitch, energy, timbre.
- Notebook hiện tại chưa cần mức phức tạp này để chứng minh domain robustness.

### 6.3 Test-time adaptation đầy đủ

**Để future work**, nhưng có thể làm bản đơn giản bằng smoothing + baseline.

TTA thích nghi mô hình lúc inference bằng unlabeled target data. Nó hợp với realtime, vì mỗi người thuyết trình là một target domain mới. Tuy nhiên SER có cảm xúc mơ hồ, pseudo-label dễ sai.

Dẫn chứng: Dong et al., **Test-Time Adaptation for Speech Emotion Recognition**, 2026. Paper: https://arxiv.org/abs/2601.16240. Repo chính thức: https://github.com/JiahengDong/SETTA

Paper này cảnh báo rằng entropy minimization và pseudo-labeling thường không ổn cho SER vì giả định "một nhãn đúng tự tin" không phù hợp với biểu hiện cảm xúc mơ hồ.

### 6.4 Source-free domain adaptation

**Để future work.**

Source-free nghĩa là khi adapt sang target, ta không còn source data, chỉ có model đã train và target unlabeled data. Hướng này thực tế vì dữ liệu speech/emotion nhạy cảm.

Dẫn chứng: Zhao et al., **Emotion-Aware Contrastive Adaptation Network for Source-Free Cross-Corpus Speech Emotion Recognition**, 2024. Paper/PDF: https://arxiv.org/abs/2401.12925 / https://arxiv.org/pdf/2401.12925

Vì sao chưa nên làm ngay:

- Notebook hiện tại vẫn có đủ 4 dataset source.
- Cần thiết kế adaptation phase riêng.
- Nên hoàn thiện domain/speaker adversarial + SupCon trước.

### 6.5 Full fine-tune WavLM/HuBERT

**Để optional/future work**, nếu có GPU mạnh.

Hiện frozen WavLM + adapter là lựa chọn hợp lý vì:

```text
nhẹ GPU
ít overfit hơn
dễ cache embedding
dễ debug
```

Full fine-tune có thể tăng accuracy, nhưng rủi ro:

```text
tốn VRAM
train lâu
dễ overfit dataset nhỏ
khó tách lỗi model với lỗi protocol
```

Nếu thử, nên bắt đầu bằng:

```text
fine-tune last 2-4 transformer layers only
lower learning rate for encoder: 1e-5 hoặc 2e-5
higher learning rate for heads/adapters: 1e-3
```

---

## 7. Kế hoạch thực nghiệm đề xuất

### Phase 1 - Chốt baseline

```text
B0: current notebook, no adversarial
B1: current notebook, domain adversarial on
```

Protocol:

```text
combined_random
combined_strict_no_tess
single-dataset
```

Metric:

```text
accuracy, macro-F1, weighted-F1, UAR, confusion matrix
```

### Phase 2 - Speaker robustness

```text
S1: add speaker head without GRL, chỉ để đo speaker leakage
S2: add speaker head + GRL, lambda_s = 0.02
S3: add speaker head + GRL, lambda_s = 0.05
```

Mục tiêu:

```text
strict macro-F1 tăng hoặc ổn định hơn
speaker accuracy giảm so với no-GRL
emotion accuracy không collapse
```

### Phase 3 - Domain + speaker adversarial

```text
D0: no adversarial
D1: domain only
D2: speaker only
D3: domain + speaker
```

Cần báo cáo:

```text
emotion macro-F1
domain accuracy
speaker accuracy
per-dataset macro-F1
```

### Phase 4 - SupCon

```text
C1: domain + speaker adversarial + SupCon 0.05
C2: domain + speaker adversarial + SupCon 0.10
```

Mục tiêu:

```text
embedding cùng emotion gần hơn
strict/cross-dataset macro-F1 tăng
confusion giữa fear/sad/neutral giảm nếu có
```

### Phase 5 - Application demo

```text
offline long audio / microphone
window 2-3s, hop 1s
emotion probability smoothing
speaker baseline
feedback text
```

---

## 8. Bảng quyết định nhanh

| Hướng | Áp dụng ngay? | Lý do | Dẫn chứng chính |
|---|---:|---|---|
| Domain adversarial / DANN | Có | Notebook đã có GRL/domain head; cần ablation để chứng minh giảm domain leakage | Ganin 2016; Abdelwahab & Busso 2018 |
| Speaker-adversarial head | Rất nên | Đánh trực tiếp vào strict speaker split thấp | Tu et al. 2019; Li et al. 2019 |
| WavLM frozen + adapter | Có | Đã có trong notebook, hợp dữ liệu nhỏ/GPU hạn chế | WavLM 2021; repo Microsoft WavLM |
| Supervised contrastive loss | Có, sau speaker head | Làm embedding cùng emotion gần nhau hơn qua speaker/domain | Khosla 2020; Xiang 2024 |
| Leave-one-dataset-out | Có | Chứng minh domain-robust thật sự | Cross-corpus SER literature |
| Diagnostics theo domain/speaker | Có | Cần để biết model overfit ở đâu | SER evaluation best practice |
| Realtime smoothing + baseline | Có | Phục vụ demo feedback thuyết trình, không cần thay model lớn | TTA/realtime motivation |
| Cross-lingual SER | Future work | Chỉ hợp khi có tiếng Việt/nhiều ngôn ngữ | Latif 2019/2022 |
| Voice conversion | Future work | Nặng, có thể làm mất emotion cues | Mote et al. 2024 |
| Test-time adaptation đầy đủ | Future work | Hợp realtime nhưng dễ nhiễu với SER | Dong et al. 2026 |
| Source-free adaptation | Future work | Thực tế nhưng cần phase adaptation riêng | ECAN 2024 |
| Full fine-tune WavLM | Optional/future | Tốn GPU, dễ overfit nếu dataset nhỏ | WavLM/wav2vec2 literature |

---

## 9. Đề xuất tên đề tài / hướng viết báo cáo

Tên tiếng Anh:

```text
Domain-Robust and Speaker-Invariant Speech Emotion Recognition for Presentation Feedback using Self-Supervised Speech Representations and Adversarial Learning
```

Tên tiếng Việt:

```text
Nhận diện cảm xúc giọng nói bền vững theo người nói và miền dữ liệu cho hệ thống phản hồi thuyết trình
```

Đoạn mô tả có thể đưa vào báo cáo:

```text
Đề tài hướng đến bài toán nhận diện cảm xúc giọng nói trong bối cảnh phản hồi thuyết trình, nơi mô hình cần tổng quát hóa tốt khi gặp người nói mới và môi trường thu âm mới. Thay vì chỉ tối ưu accuracy trên random split, nghiên cứu tập trung vào speaker-independent và domain-robust evaluation. Mô hình sử dụng pretrained speech representation như WavLM kết hợp các nhánh đặc trưng acoustic, spectrogram và statistical features. Bên cạnh emotion classifier, mô hình bổ sung domain-adversarial head và speaker-adversarial head thông qua Gradient Reversal Layer nhằm giảm phụ thuộc vào dataset và speaker identity. Ngoài ra, supervised contrastive learning được đề xuất để tăng tính phân biệt cảm xúc trong embedding space. Hệ thống cuối cùng được mở rộng bằng smoothing và speaker baseline để sinh phản hồi thuyết trình theo xu hướng giọng nói thay vì chỉ xuất nhãn cảm xúc rời rạc.
```

---

## 10. Danh sách nguồn/paper/repo đã kiểm chứng

### Nền tảng domain adversarial

1. Ganin et al. **Domain-Adversarial Training of Neural Networks**. arXiv: https://arxiv.org/abs/1505.07818. PDF JMLR: https://www.jmlr.org/papers/volume17/15-239/15-239.pdf
2. Reference DANN PyTorch implementation, không phải official paper repo: https://github.com/fungtion/DANN, https://github.com/NaJaeMin92/pytorch-DANN

### Domain adversarial cho speech emotion

3. Abdelwahab & Busso. **Domain Adversarial for Acoustic Emotion Recognition**. arXiv: https://arxiv.org/abs/1804.07690. PDF: https://lab-msp.com/MSP/publications/Abdelwahab_2018_2.pdf
4. Latif et al. **Self Supervised Adversarial Domain Adaptation for Cross-Corpus and Cross-Language Speech Emotion Recognition**. arXiv: https://arxiv.org/abs/2204.08625. PDF: https://opus.bibliothek.uni-augsburg.de/opus4/files/108969/108969.pdf

### Speaker-invariant SER

5. Tu et al. **Towards adversarial learning of speaker-invariant representation for speech emotion recognition**. arXiv: https://arxiv.org/abs/1903.09606. PDF: https://arxiv.org/pdf/1903.09606
6. Li et al. **Speaker-invariant Affective Representation Learning via Adversarial Training**. arXiv: https://arxiv.org/abs/1911.01533
7. Speaker-Invariant DANN code repo for emotion recognition: https://github.com/ihp-lab/Speaker-Invariant-Domain-Adversarial-Neural-Networks

### Self-supervised speech encoders

8. Baevski et al. **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**. arXiv: https://arxiv.org/abs/2006.11477
9. Hsu et al. **HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units**. arXiv: https://arxiv.org/abs/2106.07447. PDF: https://arxiv.org/pdf/2106.07447
10. Chen et al. **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing**. arXiv: https://arxiv.org/abs/2110.13900. PDF: https://arxiv.org/pdf/2110.13900. Official repo: https://github.com/microsoft/unilm/tree/master/wavlm
11. s3prl-ser toolkit for self-supervised speech pretraining in SER: https://github.com/bagustris/s3prl-ser

### Contrastive learning

12. Khosla et al. **Supervised Contrastive Learning**. arXiv: https://arxiv.org/abs/2004.11362. Reference repo: https://github.com/HobbitLong/SupContrast
13. Xiang. **A Cross-Corpus Speech Emotion Recognition Method Based on Supervised Contrastive Learning**. arXiv: https://arxiv.org/abs/2411.19803. PDF: https://arxiv.org/pdf/2411.19803

### Future work: source-free, voice conversion, TTA

14. Zhao et al. **Emotion-Aware Contrastive Adaptation Network for Source-Free Cross-Corpus Speech Emotion Recognition**. arXiv: https://arxiv.org/abs/2401.12925. PDF: https://arxiv.org/pdf/2401.12925
15. Mote et al. **Unsupervised Domain Adaptation for Speech Emotion Recognition using K-Nearest Neighbors Voice Conversion**. Interspeech 2024 PDF: https://www.isca-archive.org/interspeech_2024/mote24_interspeech.pdf
16. Dong et al. **Test-Time Adaptation for Speech Emotion Recognition**. arXiv: https://arxiv.org/abs/2601.16240. Official repo: https://github.com/JiahengDong/SETTA

---

## 11. Kết luận cuối cùng

Với notebook hiện tại, hướng nâng cấp thực tế nhất là:

```text
WavLM frozen + multi-branch fusion
+ domain adversarial head
+ speaker adversarial head
+ supervised contrastive loss nhẹ
+ leave-one-dataset-out evaluation
+ realtime smoothing/speaker baseline cho demo
```

Không nên cố nhồi tất cả hướng mới vào một lần. Nên chứng minh từng bước bằng ablation. Nếu strict split từ khoảng 60% tăng lên hoặc macro-F1 ổn định hơn, đồng thời domain/speaker leakage giảm, đó là đóng góp tốt hơn nhiều so với chỉ báo random accuracy cao.
