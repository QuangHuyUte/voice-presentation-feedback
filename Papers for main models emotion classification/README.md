# Papers and Code for Main Models Emotion Classification

This folder collects references for notebook `05_Main_1D_CNN_SER` and later deep SER notebooks.

## Why These References Matter

Notebook 04 already established the classical baseline with MFCC statistical features and SVM/RF/Logistic/KNN. Notebook 05 moves to frame-level sequence modeling, so the most relevant references are deep SER models that use temporal acoustic features, spectrograms, data augmentation, or self-supervised speech representations.

## Academic References

| Reference | Dataset setup | Model / idea | Result or lesson | How it informs our notebook |
|---|---|---|---|---|
| Ullah et al. as summarized in CEUR review | CREMA-D + RAVDESS + SAVEE + TESS | 1D-CNN with `ZCR + energy + entropy of energy + RMS + MFCC` | Reported 92.62% accuracy on a combined 4-dataset setup | Main reason for using 1D-CNN feature fusion in notebook 05 |
| MDPI 2022, "Recognition of Emotions in Speech Using Convolutional Neural Networks on Different Datasets" | CREMA-D + RAVDESS + SAVEE + TESS | CNN/ResNet on spectrogram-like representations | Shows that proper split can be much lower than random split | Justifies reporting both random and strict evaluation |
| Springer 2024, "Real-time speech emotion recognition using deep learning and data augmentation" | TESS, EmoDB, RAVDESS | MLP, CNN, CNN+BiLSTM with MFCC/ZCR/Mel/RMS/Chroma and augmentation | Strong reference for real-time SER and train-only augmentation | Supports augmentation, CNN+BiLSTM as later advanced model |
| emotion2vec, ACL 2024 | Large-scale self-supervised SER pretraining | Self-supervised speech emotion representation | Strong future direction for SSL embeddings | Use after 05/06 if GPU and time allow |
| CEUR 2025 hybrid SER paper | TESS + CREMA-D | LSTM/MFCC hybrid deep learning | Useful comparison for dataset-dependent performance | Helps discuss why TESS is easier than CREMA-D |

## Code References

| Repository | Local file | URL | Useful ideas |
|---|---|---|---|
| emotion2vec official code | `code/emotion2vec_README.md`, `code/emotion2vec_repo.zip` | https://github.com/ddlBoJack/emotion2vec | SSL embeddings and downstream emotion classification |
| CNN/LSTM/CLSTM four-dataset repo | `code/CNN_LSTM_CLSTM_4dataset_README.md`, `code/CNN_LSTM_CLSTM_repo.zip` | https://github.com/souradeepdutta/Speech-Emotion-Recognition-with-CNN-LSTM-CLSTM | CNN, LSTM and CLSTM pipeline on CREMA-D/RAVDESS/SAVEE/TESS |
| Emonity | `code/Emonity_README.md` | https://github.com/sv6095/Emonity | Multi-dataset SER pipeline, augmentation, MFCC/log-Mel/spectral/chroma features |
| CNN/LSTM SER example | `code/sahilkhan_CNN_LSTM_README.md` | https://github.com/sahilkhan-7/speech-emotion-recognition | Practical CNN/LSTM implementation ideas |

## Locally Downloaded Files

- `papers/CEUR_2025_SER_review_main_models.pdf`
- `papers/emotion2vec_ACL2024_arxiv.pdf`
- `code/emotion2vec_README.md`
- `code/emotion2vec_repo.zip`
- `code/CNN_LSTM_CLSTM_4dataset_README.md`
- `code/CNN_LSTM_CLSTM_repo.zip`
- `code/Emonity_README.md`
- `code/sahilkhan_CNN_LSTM_README.md`

Some publisher PDFs are protected by server-side access controls, so stable source links are kept in this README even when the PDF could not be downloaded directly.

## Ideas to Bring into Notebook 05

| Improvement | Expected benefit | Cost |
|---|---|---|
| Feature fusion: MFCC + delta + delta-delta + ZCR/RMS/energy/entropy | Closer to 1D-CNN four-dataset papers | Low |
| Train-only augmentation | Better robustness without leaking into validation/test | Low |
| Attention pooling | Lets the model weight emotional frames more strongly | Medium |
| Validation-selected best model | Avoids choosing a model based on test score | Low |
| Per-dataset metrics | Shows TESS/CREMA-D/SAVEE/RAVDESS difficulty separately | Low |
| Later 2D-CNN log-Mel | Strong spectrogram comparison model | Medium |
| Later CNN-GRU/BiLSTM | Learns longer temporal context | Medium/high |
| Later emotion2vec embedding | Strong research direction, likely improves generalization | High |
