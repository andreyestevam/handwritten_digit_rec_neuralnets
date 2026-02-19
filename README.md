# Handwritten Digit Recognition Using Neural Networks

A feedforward neural network implementation for recognizing handwritten digits (0-9) using TensorFlow/Keras.

## Overview

This project implements a classic machine learning problem in computer vision: digit recognition from handwritten images. The model achieves approximately 97% validation accuracy on the MNIST-style dataset.

## Model Architecture

- **Input Layer**: Flattens 28×28 pixel images into 784-dimensional vectors
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with softmax activation (one per digit class)

## Dataset

The dataset contains 42,000 training samples, each representing a 28×28 grayscale image:
- First column: digit label (0-9)
- Remaining 784 columns: pixel values (0-255)

Data split: 80% training, 20% validation

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
tensorflow
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook:
```bash
jupyter notebook HDR_NeuralNets.ipynb
```

The notebook covers:
1. Data loading and preprocessing
2. One-hot encoding of labels
3. Model construction and compilation
4. Training (10 epochs, batch size 32)
5. Evaluation and visualization
6. Predictions on test data

## Results

- Training accuracy: ~97%
- Validation accuracy: ~97%
- The model demonstrates good generalization with minimal overfitting

## Credits

This project was developed following the tutorial from GeeksforGeeks. Dataset and methodology courtesy of GeeksforGeeks.

## License

MIT License - see [LICENSE](LICENSE) file for details.