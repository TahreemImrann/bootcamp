# 📩 Spam SMS Classification — Day 2: Model Comparison

Bootcamp AI assignment — Day 2. Applies and compares three classification
models (SVM, Naive Bayes, Decision Tree) on a cleaned, preprocessed SMS spam
dataset, evaluating each with Accuracy, Precision, Recall, and F1 Score.

This notebook builds on **Day 1** (data cleaning, missing value handling,
duplicate removal, text cleaning, and IQR-based outlier removal), repeating
those steps here so the notebook runs standalone from a fresh Colab session.

## Dataset

**SMS Spam Collection** — 5,572 labeled SMS messages (`ham` = not spam,
`spam` = spam). Same dataset used in Day 1 (`spam.csv`).

## What this notebook does

1. **Loads and cleans the data**
   - Keeps only the label and message columns, renames them
   - Drops missing values and duplicate messages
   - Encodes labels (`ham` = 0, `spam` = 1)
   - Cleans message text (lowercase, strip digits/punctuation/extra whitespace)
   - Removes outlier messages by length using the IQR method

2. **Vectorizes the text**
   - TF-IDF vectorization (`max_features=3000`, English stop words removed)
   - Fit only on the training split, then applied to the test split (avoids
     data leakage)

3. **Trains and evaluates three classifiers** on the identical train/test
   split and TF-IDF features, for a fair comparison:
   - Support Vector Machine (linear kernel)
   - Multinomial Naive Bayes
   - Decision Tree

4. **Reports, for each model:**
   - Accuracy
   - Precision
   - Recall
   - F1 Score

5. **Compares all three models** in a summary table and a bar chart.

## How to run

1. Open the notebook in Google Colab.
2. Upload `spam.csv` via the Colab file browser (left sidebar), or clone the
   GitHub repo that contains it.
3. **Runtime → Run all.**
4. Review the printed metrics for each model and the final comparison chart.

## Tech stack

- Python
- pandas, numpy
- scikit-learn (TF-IDF, SVM, Naive Bayes, Decision Tree, evaluation metrics)
- matplotlib, seaborn (visualization)

## Results

Dataset after cleaning (missing values, duplicates, and length outliers
removed): **5,103 messages** (4,450 ham, 653 spam).

| Model         | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|-----------|--------|----------|
| SVM           | 0.9785   | 0.9658    | 0.8626 | 0.9113   |
| Naive Bayes   | 0.9706   | 0.9903    | 0.7786 | 0.8718   |
| Decision Tree | 0.9461   | 0.7794    | 0.8092 | 0.7940   |

## Conclusion

**SVM (linear kernel) was the strongest model overall**, achieving the
highest accuracy and F1 score, and the best recall of the three — meaning it
caught the most actual spam messages while still keeping precision high.

**Naive Bayes** had the highest precision (99%): when it predicts "spam,"
it's almost always correct. However, its recall was the lowest of the three,
meaning it missed more real spam messages than SVM did — a useful trade-off
to be aware of when precision matters more than catching every spam message.

**Decision Tree** underperformed on every metric compared to SVM and Naive
Bayes, consistent with tree-based models being more prone to overfitting on
high-dimensional, sparse TF-IDF text features than linear or probabilistic
models.

Overall, **SVM offered the best balance of precision and recall** for this
SMS spam detection task, making it the recommended model of the three
evaluated.

## Files

- `spam_classification_day2.ipynb` — Day 2 notebook (this task)
- `spam_classification.ipynb` — Day 1 notebook (data cleaning & EDA)
- `spam.csv` — dataset

## Author

BS Computer Science student — Bootcamp AI, Day 2 assignment.
