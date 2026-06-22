# Emonity: Speech Emotion Recognition with MFCC and Deep Learning

This project implements a robust Speech Emotion Recognition (SER) system using Mel-Frequency Cepstral Coefficients (MFCC) and advanced deep learning models in PyTorch. It leverages multiple public emotional speech datasets, extensive data augmentation, and ensemble learning to achieve high accuracy and real-time inference.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Real-Time Inference](#real-time-inference)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [References](#references)
- [License](#license)

---

## Overview

This repository provides a complete pipeline for recognizing emotions from speech audio using MFCC and other audio features, with models built in PyTorch. The system supports:
- Data loading and preprocessing from several popular datasets
- Advanced feature extraction (MFCC, log-mel, spectral, chroma, ZCR, RMS)
- Data augmentation (noise, pitch, speed, etc.)
- Training of multiple deep learning models (1D CNN, 2D CNN, CNN-BiLSTM)
- Ensemble learning for improved accuracy
- Real-time inference pipeline

## Features

- **Multi-dataset support:** CREMA-D, RAVDESS, TESS, SAVEE
- **Advanced feature extraction:** MFCCs (with deltas), log-mel spectrograms, spectral/chroma features
- **Data augmentation:** Noise injection, pitch shift, speed change, etc.
- **Deep learning models:**
  - 1D CNN with self-attention
  - 2D CNN (ResNet-like)
  - CNN-BiLSTM with attention
- **Ensemble model:** Combines all models for best performance
- **Real-time inference:** Fast, low-latency prediction pipeline
- **Comprehensive evaluation:** Accuracy, precision, recall, F1-score, confusion matrix

## Datasets

The following datasets are used (see `dataset/` directory):

- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [RAVDESS](https://zenodo.org/record/1188976)
- [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)
- [SAVEE](https://kahlan.eps.surrey.ac.uk/savee/)

**Dataset Structure Example:**
```
dataset/
  cremad/AudioWAV/
  ravdess-emotional-speech-audio/audio_speech_actors_01-24/
  surrey-audiovisual-expressed-emotion-savee/ALL/
  toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/
```
**Note:** Download and extract datasets as per their respective licenses. The notebook expects the above structure.

## Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd emonity
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Or, install manually:
   ```bash
   pip install pandas numpy librosa seaborn matplotlib scikit-learn ipython torch torchaudio xgboost lightgbm scikit-image torchvision
   ```
   For CUDA support, install PyTorch as per your GPU:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

**requirements.txt** (recommended content):
```
pandas
numpy
librosa
seaborn
matplotlib
scikit-learn
ipython
torch
torchaudio
xgboost
lightgbm
scikit-image
torchvision
```

## Usage

### Data Preparation

- Place all datasets in the `dataset/` directory as structured above.
- The notebook (`Emonity.ipynb`) will automatically process and combine the datasets, extract features, and perform augmentation.

### Training

- Open and run `Emonity.ipynb` in Jupyter or VSCode.
- The notebook will:
  - Load and preprocess data
  - Extract features and augment data
  - Split data into train/val/test
  - Train three models: 1D CNN, 2D CNN, CNN-BiLSTM
  - Save trained models as `.pth` files

**Example:**
```python
# In Emonity.ipynb
model_1d_cnn, history_1d_cnn = train_model_advanced(
    model_1d_cnn, train_loader_1d, val_loader_1d, 
    num_epochs=150, learning_rate=0.0005, weight_decay=0.01
)
torch.save(model_1d_cnn.state_dict(), 'enhanced_1d_cnn_model.pth')
```

### Evaluation
<img width="1990" height="1190" alt="download (7)" src="https://github.com/user-attachments/assets/24f5b6a7-a9c0-49f1-aea9-36b07df64b12" />

- <img width="742" height="590" alt="download (8)" src="https://github.com/user-attachments/assets/6f7e22b8-abbd-4ce2-8aa9-893d620b31e5" />
The notebook evaluates each model and the ensemble on the test set, reporting accuracy, precision, recall, F1-score, and confusion matrices.
- **Best results:**
  - **1D CNN:** Accuracy: 0.705, Precision: 0.712, Recall: 0.705, F1: 0.698
  - **2D CNN:** Accuracy: 0.901, Precision: 0.903, Recall: 0.901, F1: 0.901
  - **CNN-BiLSTM:** Accuracy: 0.794, Precision: 0.796, Recall: 0.794, F1: 0.793
  - **Ensemble:** Accuracy: 0.885, Precision: 0.888, Recall: 0.885, F1: 0.885

| Model         | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| 1D CNN       | 0.705    | 0.712     | 0.705  | 0.698    |
| 2D CNN       | 0.901    | 0.903     | 0.901  | 0.901    |
| CNN-BiLSTM   | 0.794    | 0.796     | 0.794  | 0.793    |
| Ensemble     | 0.885    | 0.888     | 0.885  | 0.885    |

- See the notebook for per-class metrics and confusion matrices.

### Real-Time Inference

- The notebook provides a `RealTimeEmotionClassifier` class for low-latency emotion prediction from audio files.

**Example usage:**
```python
from SpeechEr import RealTimeEmotionClassifier
classifier = RealTimeEmotionClassifier('speech_emotion_ensemble_model.pth')
emotion, confidence, probabilities, inference_time = classifier.predict_emotion('path/to/audio.wav')
print(f"Predicted: {emotion} (confidence: {confidence:.2f}, time: {inference_time:.3f}s)")
```

## Model Architectures

- **Enhanced 1D CNN:** Self-attention, batch norm, dropout, global pooling
- **Enhanced 2D CNN:** ResNet-like blocks, batch norm, dropout, global pooling
- **Enhanced CNN-BiLSTM:** CNN feature extractor, BiLSTM, attention, dense layers
- **Ensemble:** Weighted average of all models' predictions

## Results

### Overall Metrics

| Model         | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| 1D CNN       | 0.705    | 0.712     | 0.705  | 0.698    |
| 2D CNN       | 0.901    | 0.903     | 0.901  | 0.901    |
| CNN-BiLSTM   | 0.794    | 0.796     | 0.794  | 0.793    |
| Ensemble     | 0.885    | 0.888     | 0.885  | 0.885    |

### Per-Class Metrics

#### Enhanced 1D CNN
| Class     | Precision | Recall  | F1-Score |
|-----------|-----------|---------|----------|
| angry     | 0.7301    | 0.8567  | 0.7883   |
| disgust   | 0.6941    | 0.6317  | 0.6614   |
| fear      | 0.7878    | 0.4950  | 0.6080   |
| happy     | 0.7238    | 0.5633  | 0.6336   |
| neutral   | 0.6381    | 0.8433  | 0.7265   |
| sad       | 0.6294    | 0.6850  | 0.6560   |
| surprise  | 0.8186    | 0.9439  | 0.8768   |

#### Enhanced 2D CNN
| Class     | Precision | Recall  | F1-Score |
|-----------|-----------|---------|----------|
| angry     | 0.9401    | 0.9150  | 0.9274   |
| disgust   | 0.8887    | 0.8783  | 0.8835   |
| fear      | 0.9325    | 0.8517  | 0.8902   |
| happy     | 0.9201    | 0.8833  | 0.9014   |
| neutral   | 0.9012    | 0.9117  | 0.9064   |
| sad       | 0.8174    | 0.9100  | 0.8612   |
| surprise  | 0.9303    | 0.9872  | 0.9579   |

#### Enhanced CNN-BiLSTM
| Class     | Precision | Recall  | F1-Score |
|-----------|-----------|---------|----------|
| angry     | 0.8495    | 0.8750  | 0.8621   |
| disgust   | 0.8081    | 0.7300  | 0.7671   |
| fear      | 0.7797    | 0.6783  | 0.7255   |
| happy     | 0.8027    | 0.6917  | 0.7431   |
| neutral   | 0.7576    | 0.8750  | 0.8121   |
| sad       | 0.7038    | 0.8000  | 0.7488   |
| surprise  | 0.9115    | 0.9719  | 0.9407   |

#### Ensemble Model
| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.8850  |
| Precision  | 0.8875  |
| Recall     | 0.8850  |
| F1-Score   | 0.8845  |

See the notebook for confusion matrices and further analysis.

## Troubleshooting

- **CUDA not available:** Ensure you have a compatible GPU and the correct PyTorch version installed.
- **Dataset not found:** Double-check dataset paths and structure.
- **Out of memory:** Reduce batch size or use a machine with more RAM/GPU memory.
- **Librosa or torchaudio errors:** Ensure all dependencies are installed and up to date.

## References

- [CREMA-D Dataset](https://github.com/CheyneyComputerScience/CREMA-D)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [SAVEE Dataset](https://kahlan.eps.surrey.ac.uk/savee/)
- [TESS Dataset](https://tspace.library.utoronto.ca/handle/1807/24487)

## License

This project is for academic and research purposes. Please check individual dataset licenses for usage restrictions.
