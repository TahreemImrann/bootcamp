# Full Fine-Tuning vs LoRA on BERT

A practical comparison of **Full Fine-Tuning** and **LoRA (Low-Rank Adaptation)** for sentiment classification using `bert-base-uncased` and the IMDb dataset.

## 📌 Project Overview

This project demonstrates and compares two approaches for adapting a pretrained BERT model to a downstream text-classification task:

1. **Full Fine-Tuning** — all BERT parameters are updated during training.
2. **LoRA (PEFT)** — the original BERT weights remain largely frozen while small trainable low-rank adapter matrices are added.

The experiment uses the **IMDb movie-review sentiment dataset** and evaluates both approaches using accuracy, training time, trainable parameters, and GPU memory consumption.

## 🎯 Objectives

* Fine-tune `bert-base-uncased` for binary sentiment classification.
* Compare traditional full-model fine-tuning with LoRA.
* Measure the number of trainable parameters.
* Compare training time and GPU memory usage.
* Evaluate classification accuracy.
* Test both trained models on a sample movie review.

## 🧰 Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* LoRA
* BERT
* Google Colab / NVIDIA T4 GPU
* Jupyter Notebook

The notebook installs and uses `torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, and `peft`.

## 📊 Dataset

The experiment uses the **Stanford IMDb dataset**:

```python
dataset = load_dataset("stanfordnlp/imdb")
```

The complete dataset contains:

* 25,000 training examples
* 25,000 test examples
* 50,000 unsupervised examples

For the experiment, a smaller subset was selected:

* **2,000 training samples**
* **500 test samples**

Both subsets were shuffled using `seed=42`.

## 🔤 Tokenization

The `bert-base-uncased` tokenizer is used.

Reviews are tokenized with:

* `padding="max_length"`
* `truncation=True`
* `max_length=256`

The label column is renamed from `label` to `labels` for compatibility with the Transformers training workflow.

## 🧠 Model

The base model is:

```text
bert-base-uncased
```

A sequence-classification head with two labels is used:

```python
AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

The two classes represent:

* `0` → Negative
* `1` → Positive

## 🔥 Full Fine-Tuning

For full fine-tuning, every model parameter is made trainable:

```python
for param in model.parameters():
    param.requires_grad = True
```

The experiment trains for **2 epochs** with a batch size of **8**.

### Full Fine-Tuning Results

| Metric               |      Result |
| -------------------- | ----------: |
| Total Parameters     | 109,483,778 |
| Trainable Parameters | 109,483,778 |
| Trainable Percentage |     100.00% |
| Training Time        |  236.42 sec |
| Peak GPU Memory      |  2673.18 MB |
| Evaluation Accuracy  |      87.60% |

The final full fine-tuning evaluation produced an accuracy of **87.6%**.

## ⚡ LoRA Fine-Tuning

LoRA is implemented using the Hugging Face **PEFT** library.

The LoRA configuration used in the experiment is:

```python
LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)
```

The LoRA adapters target the BERT **query** and **value** modules.

The experiment uses a learning rate of `1e-3`, with:

* Batch size: 8
* Epochs: 2
* Evaluation: every epoch
* Logging: every 50 steps

### LoRA Results

| Metric               |      Result |
| -------------------- | ----------: |
| Total Parameters     | 110,075,140 |
| Trainable Parameters |     591,362 |
| Trainable Percentage |       0.54% |
| Training Time        |  180.70 sec |
| Peak GPU Memory      |  2863.51 MB |
| Evaluation Accuracy  |      89.00% |

The LoRA experiment achieved **89.0% evaluation accuracy**.

## 🏆 Final Comparison

| Metric               | Full Fine-Tuning | LoRA (PEFT) |
| -------------------- | ---------------: | ----------: |
| Total Parameters     |      109,483,778 | 110,075,140 |
| Trainable Parameters |      109,483,778 | **591,362** |
| Trainable %          |          100.00% |   **0.54%** |
| Training Time        |          236.42s | **180.70s** |
| Peak GPU Memory      |   **2673.18 MB** |  2863.51 MB |
| Evaluation Accuracy  |           87.60% |  **89.00%** |

These values are the final comparison reported by the notebook.

## 📈 Key Observations

### 1. LoRA dramatically reduces trainable parameters

Full fine-tuning updates approximately **109.5 million parameters**, whereas LoRA trains only **591,362 parameters**.

That means only **0.54%** of the LoRA model's parameters are trainable.

### 2. LoRA was faster in this experiment

The recorded training time was:

```text
Full Fine-Tuning: 236.42 seconds
LoRA:             180.70 seconds
```

LoRA completed the training run about **56 seconds faster** in this experiment.

### 3. LoRA achieved slightly higher accuracy

The recorded evaluation accuracy was:

```text
Full Fine-Tuning: 87.60%
LoRA:             89.00%
```

Therefore, LoRA achieved a **1.4 percentage-point improvement** on the selected evaluation subset in this particular run.

### 4. GPU memory was not lower in this particular run

Interestingly, the measured peak GPU memory was:

```text
Full Fine-Tuning: 2673.18 MB
LoRA:             2863.51 MB
```

So although LoRA substantially reduced the number of trainable parameters, it did **not** produce lower peak GPU memory in this specific notebook run.

## 🧪 Sample Prediction

The notebook tests the trained model using:

```text
"This movie was surprisingly good, I loved it."
```

The full fine-tuned model predicts:

```text
Positive
```

The LoRA model also predicts:

```text
Positive
```

The notebook therefore demonstrates successful inference from both approaches.

## 📁 Project Structure

```text
.
├── fullfinetune_vs_LORA_Task07.ipynb
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install dependencies

```bash
pip install torch transformers datasets accelerate evaluate peft
```

### 3. Open the notebook

```bash
jupyter notebook fullfinetune_vs_LORA_Task07.ipynb
```

Alternatively, upload the notebook to **Google Colab** and run it with a GPU runtime.

### 4. Run the notebook

Run the cells sequentially to:

1. Install dependencies
2. Load the IMDb dataset
3. Load the BERT tokenizer
4. Tokenize the reviews
5. Create the training/test subsets
6. Perform full fine-tuning
7. Perform LoRA fine-tuning
8. Compare the results
9. Run sample predictions

## 💡 Conclusion

This experiment demonstrates that **LoRA can adapt BERT using a tiny fraction of the trainable parameters while achieving competitive—and in this run slightly better—evaluation accuracy than full fine-tuning**.

The main result is the large reduction in trainable parameters:

> **100.00% → 0.54%**

while evaluation accuracy increased from:

> **87.60% → 89.00%**

and recorded training time decreased from:

> **236.42s → 180.70s**

However, the measured peak GPU memory was higher for LoRA in this particular run, so the experiment should not be interpreted as showing that LoRA always reduces every resource metric.

## 📚 Notebook

The complete implementation and experiment are available in:

`fullfinetune_vs_LORA_Task07.ipynb`

---

**Task:** Task 07 — Full Fine-Tuning vs LoRA
**Base Model:** `bert-base-uncased`
**Dataset:** IMDb Sentiment Classification
**Framework:** PyTorch + Hugging Face Transformers + PEFT
**Runtime:** Google Colab / NVIDIA T4 GPU
