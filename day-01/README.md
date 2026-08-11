# Day 01 - Spam SMS Classification

## Overview

This is my Day 01 task for the 10-Day Bootcamp.

The task focuses on preprocessing and analyzing an SMS spam dataset, detecting and removing outliers, visualizing the data, and building a machine learning classification model to compare performance before and after outlier removal.

## Dataset

The task uses the **SMS Spam Collection** dataset containing 5,572 labeled SMS messages.

The messages are classified into two categories:

- `ham` - Not spam
- `spam` - Spam

## Objectives

The main objectives of this task were to:

- Load and inspect the dataset
- Clean and preprocess the data
- Analyze missing values
- Remove duplicate records
- Clean and normalize text data
- Encode the target labels
- Perform feature engineering
- Visualize the dataset
- Detect outliers using the IQR method
- Remove outliers
- Build a spam classification model before outlier removal
- Build the same classification model after outlier removal
- Compare the model performance before and after outlier removal

## Technologies & Libraries Used

- Python
- Google Colab / Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- Scikit-learn

## Data Preprocessing

The dataset was inspected and unnecessary columns were removed. The remaining columns were renamed to:

- `label`
- `message`

Missing values were checked, and duplicate records were removed. The notebook also converts the `ham` and `spam` labels into numerical values:

- `ham` → `0`
- `spam` → `1`

The SMS text was cleaned by:

- Converting text to lowercase
- Removing numbers
- Removing punctuation
- Removing extra whitespace

## Feature Engineering

Two numerical features were created from the SMS messages:

- **Message length** - number of characters in each message
- **Word count** - number of words in each message

These features were used for analysis and outlier detection.

## Data Visualization

Several visualizations were created to understand the dataset, including:

- Ham vs. spam class distribution
- Message length distribution
- Word count distribution
- Boxplot of message length before outlier removal
- Word clouds for ham and spam messages
- Correlation heatmap

## Outlier Detection

The **Interquartile Range (IQR)** method was used to detect outliers based on message length.

Messages outside the calculated lower and upper IQR bounds were considered outliers and removed from the dataset.

A boxplot was also created after outlier removal to visualize the cleaned distribution.

## Machine Learning Model

A text classification pipeline was created using:

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the cleaned SMS text into numerical features.

### Multinomial Naive Bayes

A **Multinomial Naive Bayes** classifier was used as the classification model.

The model was trained and evaluated twice:

1. Before outlier removal
2. After outlier removal

The following evaluation metrics were used:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Before vs. After Comparison

The model performance before and after outlier removal was compared using the same TF-IDF and Multinomial Naive Bayes pipeline.

A comparison table and visualization were created to observe how removing outliers affected the classification performance.

## Key Learnings

Through this task, I learned about:

- Data preprocessing and cleaning
- Handling missing values and duplicates
- Text preprocessing
- Label encoding
- Feature engineering
- Data visualization
- Outlier detection using the IQR method
- TF-IDF vectorization
- Multinomial Naive Bayes classification
- Evaluating machine learning models
- Comparing model performance before and after data cleaning

## File

- `spam_classification.ipynb` - Jupyter/Google Colab notebook containing the complete Day 01 task.

## Conclusion

This task demonstrated a complete basic machine learning workflow for SMS spam classification, starting from data preprocessing and exploratory analysis through feature engineering, outlier detection, model training, and performance comparison.

The notebook also demonstrates how data cleaning and outlier removal can affect the dataset and the resulting machine learning model.

## Future Improvements

Possible improvements include:

- Trying other classification algorithms such as Logistic Regression or SVM
- Using n-grams with TF-IDF
- Applying stemming or lemmatization
- Experimenting with other outlier detection methods such as Z-score
