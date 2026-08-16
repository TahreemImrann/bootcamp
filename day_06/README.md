# Task 06 — BERT Sentiment Analysis

This project fine-tunes **BERT-base-uncased** for binary sentiment classification using the IMDb movie-review dataset.

## Objective

Build a sentiment-analysis pipeline that:

1. Loads the IMDb dataset
2. Tokenizes movie reviews with BERT's tokenizer
3. Fine-tunes BERT for two-class classification
4. Evaluates the model using accuracy
5. Uses the trained model to classify a new review

## Dataset

The notebook loads:

`stanfordnlp/imdb`

For a lightweight training run, it uses:

- `2,000` shuffled training examples
- `500` shuffled test examples

The shuffle uses seed `42`.

## Preprocessing

The `bert-base-uncased` tokenizer is used.

Reviews are:

- Tokenized
- Padded to a fixed length
- Truncated when necessary
- Limited to a maximum sequence length of `256`

The dataset label is renamed to `labels` for compatibility with the Hugging Face Trainer.

## Model

The notebook uses:

`bert-base-uncased`

with:

- Number of labels: `2`
- Class `0`: Negative
- Class `1`: Positive

The model is loaded with `AutoModelForSequenceClassification`.

## Training

Training is performed using the Hugging Face `Trainer` API.

Configuration in the notebook:

- Training batch size: `8`
- Evaluation batch size: `8`
- Epochs: `2`
- Logging every `50` steps
- Evaluation: once per epoch
- Reporting: disabled

## Evaluation

The notebook defines an accuracy metric by comparing the predicted class with the ground-truth label.

The trained model is evaluated on the selected test subset.

## Example Prediction

The notebook tests the sentence:

> “This movie was surprisingly good, I loved it.”

The model prints either:

- `Positive`
- `Negative`

based on the predicted class.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- Evaluate
- Matplotlib

## Files

- `bert_model_task06.ipynb` — complete BERT fine-tuning and sentiment prediction workflow

## How to Run

Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab and run the cells from top to bottom.

An internet connection is required when downloading the IMDb dataset, tokenizer, and pretrained BERT model.

## Notes

This notebook intentionally uses a reduced subset of IMDb for a faster demonstration. The training configuration is therefore suitable for experimentation and learning rather than for achieving a state-of-the-art benchmark result.
