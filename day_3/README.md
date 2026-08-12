# Neural Network Day 3 --- MNIST Classification & Activation Functions

## Overview

This project implements a simple neural network for handwritten digit
classification using the **MNIST dataset**. The task demonstrates image
preprocessing, tensor multiplication, neural-network computation,
logits, activation functions, predictions, accuracy comparison, and
confusion-matrix evaluation.

The notebook used for this task is:

`Neural_networks_day3.ipynb`

## Dataset

The project uses the MNIST dataset loaded with:

``` python
load_dataset("ylecun/mnist")
```

Dataset size:

-   **Training images:** 60,000
-   **Test images:** 10,000
-   **Image size:** 28 × 28 pixels
-   **Classes:** 10 digits (0--9)

## Preprocessing

Each MNIST image is converted into a NumPy array and preprocessed as
follows:

1.  Convert the image to `float32`.
2.  Normalize pixel values by dividing by `255.0`.
3.  Flatten each `28 × 28` image into a `784`-element vector.
4.  Convert the processed data into PyTorch tensors.

Resulting shapes:

-   `X_train`: `(60000, 784)`
-   `X_test`: `(10000, 784)`

## Tensor Multiplication

A manual tensor-multiplication example is included to demonstrate the
operation performed by a linear layer:

``` text
output = input × weightᵀ + bias
```

For the demonstration:

-   Input shape: `(1, 784)`
-   Weight shape: `(10, 784)`
-   Output shape: `(1, 10)`

This shows how an input image is transformed into values for the output
neurons.

## Neural Network Architecture

The neural network is a fully connected classifier:

``` text
784 Input Features
       ↓
Linear Layer: 784 → 128
       ↓
ReLU
       ↓
Linear Layer: 128 → 64
       ↓
ReLU
       ↓
Linear Layer: 64 → 10
       ↓
Raw Logits
```

### Activation Functions

**ReLU** is used in the internal/hidden layers:

``` text
Linear → ReLU → Linear → ReLU → Linear
```

ReLU provides non-linearity and allows the hidden layers to learn useful
patterns from the images.

The final output layer produces **10 raw logits**, one for each digit
class.

> **Important:** Softmax and Sigmoid are not applied sequentially. They
> are tested separately on the same final raw logits.

The comparison is:

``` text
                 Raw Logits
                /          \
           Softmax         Sigmoid
              ↓               ↓
        Predictions      Predictions
```

## Model Training

The model is trained using:

-   **Loss function:** CrossEntropyLoss
-   **Optimizer:** Adam
-   **Learning rate:** `0.001`
-   **Epochs:** `5`
-   **Batch size:** `128`

Training results:

  Epoch       Loss   Training Accuracy
  ------- -------- -------------------
  1         0.4162              88.89%
  2         0.1716              94.98%
  3         0.1206              96.38%
  4         0.0920              97.20%
  5         0.0731              97.77%

## Logits Before Activation

After training, the model produces raw logits before applying an output
activation.

The test output has the shape:

``` text
(10000, 10)
```

Each test image therefore receives 10 output scores corresponding to the
10 digit classes.

## Softmax Results

Softmax is applied to the raw logits:

``` text
Raw Logits → Softmax → Probabilities
```

Softmax produces normalized values whose sum is approximately `1`.

### Result

**Softmax Accuracy: 97.33%**

## Sigmoid Results

Sigmoid is also applied to the same raw logits:

``` text
Raw Logits → Sigmoid → Output Scores
```

Sigmoid maps each output independently to a value between `0` and `1`.
Unlike Softmax, the output values are not required to sum to `1`.

### Result

**Sigmoid Accuracy: 97.33%**

## Softmax vs Sigmoid Comparison

  Metric                     Softmax   Sigmoid
  ------------------------ --------- ---------
  Accuracy                    97.33%    97.33%
  Prediction agreement          100%      100%
  Output values sum to 1         Yes        No

In this experiment, Softmax and Sigmoid produced exactly the same
predicted classes.

This happens because both transformations preserve the ordering of the
logits, so selecting the largest output with `argmax` results in the
same predicted class.

However, the numerical output values have different meanings:

-   **Softmax:** normalized distribution across the 10 classes.
-   **Sigmoid:** independent score for each class.

## Confusion Matrix

A confusion matrix is generated for the Softmax predictions.

The diagonal entries represent correctly classified images, while
off-diagonal entries represent misclassifications.

The notebook also compares the Softmax and Sigmoid predictions. Because
the predictions match 100% in this experiment, their confusion matrices
are identical.

## Final Conclusion

This task demonstrates the complete flow of a basic neural-network
image-classification pipeline:

``` text
MNIST Images
     ↓
Preprocessing
     ↓
Flattened Tensor
     ↓
Tensor Multiplication / Linear Layers
     ↓
ReLU in Hidden Layers
     ↓
Raw Logits
     ↓
 ┌───────────┬───────────┐
 ↓           ↓
Softmax    Sigmoid
 ↓           ↓
Predictions
 ↓           ↓
Accuracy & Confusion Matrix
```

The model successfully classified MNIST handwritten digits with **97.33%
test accuracy** using both Softmax and Sigmoid for prediction in this
experiment. The predictions from both approaches matched **100%**.

The key learning is that the three activation functions have different
roles:

-   **ReLU** is used inside the neural network's hidden layers to
    introduce non-linearity.
-   **Softmax** is appropriate for multiclass classification because it
    produces a normalized probability distribution across classes.
-   **Sigmoid** independently maps each output to a value between `0`
    and `1`.

Therefore, the experiment should be understood as:

**ReLU for hidden layers + Softmax OR Sigmoid for the output
comparison**, rather than applying ReLU → Sigmoid → Softmax
sequentially.

## Requirements

The notebook uses:

-   Python
-   PyTorch
-   NumPy
-   Matplotlib
-   Hugging Face Datasets
-   scikit-learn

## How to Run

1.  Open `Neural_networks_day3.ipynb` in Google Colab.
2.  Run the cells from top to bottom.
3.  The notebook downloads the MNIST dataset.
4.  The model is trained for 5 epochs.
5.  Raw logits are generated for the test set.
6.  Softmax and Sigmoid outputs are calculated separately.
7.  Predictions and accuracy are compared.
8.  The confusion matrix is displayed.

## Project Structure

``` text
.
├── Neural_networks_day3.ipynb
└── README.md
```

## Key Results

-   **Dataset:** MNIST
-   **Training samples:** 60,000
-   **Test samples:** 10,000
-   **Input features:** 784
-   **Hidden layers:** 128 and 64 neurons
-   **Hidden activation:** ReLU
-   **Output classes:** 10
-   **Softmax accuracy:** 97.33%
-   **Sigmoid accuracy:** 97.33%
-   **Softmax/Sigmoid prediction agreement:** 100%
