# Task 04 — Implement Attention Is All You Need

This project implements the main components of the Transformer architecture described in the **“Attention Is All You Need”** paper.

## Objective

Implement the core Transformer encoder-decoder architecture from scratch using PyTorch, including:

- Scaled dot-product attention
- Multi-head attention
- Sinusoidal positional encoding
- Transformer encoder layers
- Causal masking for decoder self-attention
- Transformer decoder layers
- Encoder-decoder cross-attention
- Token embeddings and output projection

## Implementation Overview

### 1. Scaled Dot-Product Attention

The notebook implements:

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ)V`

The implementation also supports an optional attention mask.

### 2. Multi-Head Attention

The `MultiHeadAttention` module:

- Projects queries, keys, and values
- Splits representations across multiple attention heads
- Applies scaled dot-product attention independently to each head
- Concatenates the heads
- Applies the final output projection

The default configuration is:

- `d_model = 512`
- `h = 8`
- `d_k = 64`

### 3. Positional Encoding

Because the Transformer does not use recurrence, sinusoidal positional encodings are added to token representations so that sequence order is represented.

### 4. Transformer Encoder

Each encoder layer contains:

1. Multi-head self-attention
2. Residual connection + layer normalization
3. Feed-forward network
4. Residual connection + layer normalization

The notebook uses 6 encoder layers by default.

### 5. Transformer Decoder

Each decoder layer contains:

1. Masked self-attention
2. Encoder-decoder cross-attention
3. Feed-forward network
4. Residual connections and layer normalization

A causal mask prevents a decoder position from attending to future target tokens.

### 6. Full Transformer

The `Transformer` class combines:

- Source and target embeddings
- Encoder
- Decoder
- Final linear projection to the target vocabulary

## Sanity Checks

The notebook runs test inputs through the encoder and full Transformer and checks the resulting tensor shapes.

For the full model, the example uses:

- Batch size: `2`
- Source sequence length: `10`
- Target sequence length: `8`
- Source vocabulary size: `1000`
- Target vocabulary size: `1000`

Expected output shape:

`[2, 8, 1000]`

## Technologies

- Python
- PyTorch
- Matplotlib
- Hugging Face Transformers/Datasets packages

## Files

- `attention_task04.ipynb` — complete implementation and sanity tests

## How to Run

Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab and run the cells from top to bottom.

The notebook installs the required Python packages in its first cell.
