# Experiment Log

## Experiment 01 — CPU Development Run

### Model
DistilBERT (`distilbert-base-uncased`)

### Dataset
6,000 training reviews  
1,500 validation reviews  
1,500 test reviews

### Configuration
- Epochs: 1
- Batch size: 4
- Maximum sequence length: 128
- Learning rate: 2e-5
- Weight decay: 0.01
- Device: CPU

### Results
- Validation Accuracy: 79.2%
- Validation Weighted F1: 78.86%
- Validation Macro F1: 75.0%

### Class-wise F1
- Negative: 0.82
- Neutral: 0.54
- Positive: 0.88

### Observations
- DistilBERT successfully learned sentiment classification.
- Positive and Negative classification performed strongly.
- Neutral sentiment was more difficult to identify.
- CPU training is significantly slower than GPU training.

### Next Step
Train the final model using the larger review dataset in a GPU environment.