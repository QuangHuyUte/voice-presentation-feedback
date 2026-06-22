# Papers and Code for Emotion Classification

This folder collects references for the baseline emotion classification stage.

## Papers / Academic References

| Reference | Dataset setup | Method / idea | Result to compare |
|---|---|---|---|
| Novais et al., "Emotion Classification from Speech by an Ensemble Strategy" | RAVDESS + CREMA-D + SAVEE + TESS | Aggregator / ensemble over primary classifiers | ACM page reports all-dataset setup and accuracies around 62-65%; related abstract reports 75.56% on RAVDESS and 86.43% on RAVDESS + SAVEE + TESS |
| "A Framework for Emotion and Sentiment Predicting Supported in Speech, Face and Text Analysis" | RAVDESS, TESS, CREMA-D, SAVEE | Framework with multiple speech emotion classifiers | Notes four-dataset SER experiments and best result above 86% for a 3-of-4 combination |
| Wiley review, "Speech Emotion Recognition Using Transfer Learning: A ..." | Multi-corpus RAVDESS + TESS + SAVEE + CREMA-D, 6 classes | SVM + majority voting with feature selection | Reported 92.55% accuracy in review table |
| Lee & Nadeem, "Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention" | TESS, RAVDESS, CREMA-D, SAVEE, EMO-DB, EMOVO | MFCC + augmentation + 1D-CNN attention | Per-dataset results: RAVDESS 99.23%, TESS 99.82%, CREMA-D 89.31%, SAVEE 97.49% |
| Mohan et al., "Speech Emotion Classification using Ensemble Models with MFCC" | SER ensemble reference | MFCC + ensemble models | Useful design reference for ensemble baselines |

## Code References

| Repository | URL | Useful ideas |
|---|---|---|
| Emonity | https://github.com/sv6095/Emonity | Multi-dataset support for CREMA-D, RAVDESS, TESS, SAVEE; MFCC/log-Mel/spectral/chroma features; augmentation; model ensemble |
| deep-emotion-recognition | https://github.com/miladranaeisiadat/deep-emotion-recognition | Multi-corpus SER pipeline using MFCC features and neural models; includes RAVDESS, CREMA-D, TESS, SAVEE among other datasets |

## Locally Downloaded Files

- `papers/Detection_classification_SER_TESS_CREMA_review.pdf`
- `code/Emonity.ipynb`

Some publisher PDFs and full GitHub archive downloads were blocked or too slow in this environment, so the table above keeps stable source links for manual reference. The downloaded `Emonity.ipynb` is included as a compact code reference because it already contains a multi-dataset SER workflow for CREMA-D, RAVDESS, TESS and SAVEE.
