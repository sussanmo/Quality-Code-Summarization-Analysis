# Training Configuration for Reproducibility

- Epochs: 10
- Learning rate: 1e-3
- Optimizer: Adam
- Loss function: Binary Cross Entropy
- Attention heads: 4

## Embedding Dimensions
- Base embedding hidden dimension: 32
- Combined embedding hidden dimension: 160
  - Method token embeddings: 32
  - Semantic category embeddings: 64
  - Fixation duration embeddings: 64

## Training Setup
- Batch size: 32
- Dropout: 0.02
- Early stopping patience: 3
- Random seeds: [0, 1, 42, 123, 12345]
- Metrics: Accuracy, Precision, Recall, F1 Score

## References
- [Dropout](https://doi.org/10.5555/2627435.2670313)
- [Adam Optimizer](https://arxiv.org/abs/1412.6980)
- [Binary Cross Entropy](https://proceedings.neurips.cc/paper_files/paper/2020/file/1e14bfeab435b4ebebc314e4a62c32dc-Paper.pdf)
