# Emonity Reference Note

Source repo: https://github.com/sv6095/Emonity

This folder keeps the lightweight reference files only. The original repository included a `dataset/` directory and trained `.pth` model weights; those were removed here because the project already has its own processed dataset and model outputs.

## What the code uses

- Datasets: CREMA-D, RAVDESS, TESS, SAVEE.
- Labels in the original README/notebook: 7 emotions, including `surprise`.
- Main feature groups:
  - MFCC with delta and delta-delta features.
  - Log-Mel spectrogram.
  - Spectral features.
  - Chroma.
  - Zero-crossing rate.
  - RMS energy.
- Augmentation:
  - Noise injection.
  - Pitch shifting.
  - Time stretching.
- Split protocol found in the notebook:
  - Stratified random train/test split: `test_size=0.2`.
  - Stratified random train/validation split from the train portion: `test_size=0.25`.
  - Effective split is about 60% train, 20% validation, 20% test.
  - No speaker-aware or speaker-independent split was found in the notebook.
- Models:
  - Enhanced 1D CNN with self-attention.
  - Enhanced 2D CNN with residual blocks on Log-Mel input.
  - Enhanced CNN-BiLSTM with attention.
  - Weighted ensemble using validation accuracy as model weights.

## How to compare fairly in this project

When adapting Emonity to this project, do not compare its README accuracy directly with strict speaker-aware results. Re-train the architecture on the project's own split protocols:

- `combined_random`: closest to Emonity's random sample split.
- `combined_strict_no_tess`: stricter generalization test with no speaker overlap, excluding TESS.
- `single-dataset`: train/test inside each dataset to understand corpus-specific performance.

The project uses 6 common emotions: `neutral`, `happy`, `sad`, `angry`, `fear`, `disgust`. Therefore, the final classifier layer should output 6 classes instead of the original 7-class setup.
