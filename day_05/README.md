# Task 05 — RNN vs LSTM: Vanishing Gradient Problem

This project demonstrates the **vanishing gradient problem** in recurrent neural networks and compares a Vanilla RNN with an LSTM.

## Objective

The notebook investigates how gradients behave across timesteps when a loss is propagated backward from the final timestep.

The comparison is between:

- Vanilla RNN
- LSTM

## Implementation Overview

### 1. Vanilla RNN

The `VanillaRNN` class implements recurrent hidden-state updates using:

- An input-to-hidden linear transformation
- A hidden-to-hidden linear transformation
- `tanh` activation

The hidden state is stored at every timestep and its gradient is retained for analysis.

### 2. LSTM

The `SimpleLSTM` class uses PyTorch's `LSTMCell`.

It maintains:

- Hidden state `h`
- Cell state `c`

The hidden state at every timestep is retained so its gradient can be measured after backpropagation.

### 3. Measuring Gradient Flow

A sequence of length `50` is generated with:

- Batch size: `4`
- Input size: `20`
- Hidden size: `32`

The loss is defined as the sum of the hidden state at the **last timestep**.

After backpropagation, the notebook calculates the gradient norm at every timestep.

### 4. Visualization

The notebook plots gradient norms on a logarithmic scale for both models.

The resulting graph is titled:

**Vanishing Gradient: RNN vs LSTM**

This makes it possible to visually compare how effectively gradients are propagated through the sequence.

## Expected Interpretation

Vanilla RNNs can suffer from rapidly decreasing gradients as the distance from the final timestep increases. This makes learning long-range dependencies difficult.

LSTMs introduce a separate cell state and gating mechanism that can help preserve information and improve gradient flow over longer sequences.

The notebook demonstrates this behavior experimentally rather than relying only on a theoretical explanation.

## Technologies

- Python
- PyTorch
- Matplotlib

## Files

- `rnn_lstm_task05.ipynb` — RNN/LSTM implementation, gradient analysis, and visualization
