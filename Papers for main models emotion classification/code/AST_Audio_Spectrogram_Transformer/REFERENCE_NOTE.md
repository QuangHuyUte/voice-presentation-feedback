# AST Reference Note

This folder stores reference material for the audio-pretrained spectrogram branch used in the 05 multi-representation ensemble notebook.

- Paper: Audio Spectrogram Transformer, arXiv:2104.01778
- Official code repository: https://github.com/YuanGongND/ast
- Default HuggingFace checkpoint used in the notebook: `MIT/ast-finetuned-audioset-10-10-0.4593`

Project adaptation:

```text
audio 16 kHz
-> pretrained spectrogram encoder embeddings
-> small classifier head
-> validation-weighted ensemble with temporal and statistical branches
```

The notebook presents the system as a 3-branch architecture rather than naming the whole method after AST.
