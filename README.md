# Financial Sentiment Analysis of Executive Letters

## ✨ Overview

This project applies various machine learning approaches to perform sentiment analysis on executive letters extracted from Danish annual financial reports. We benchmarked classical ML models against modern transformer-based architectures, with a special focus on FinBERT due to its domain specificity in financial text.

The models were trained on a combination of:

* Financial Phrase Bank (FPB)
* FiQA Dataset
* Custom-labeled sentences from executive letters of 20 leading Danish firms

Our results demonstrate that FinBERT delivers the highest accuracy and is most suitable for detecting nuanced sentiment in financial disclosures.

---

## 🚀 Importance of Environment

The project requires precise environment configurations to ensure compatibility across ML libraries, especially when working with TensorFlow, Huggingface Transformers, and SpaCy pipelines.

> ⚡ **NOTE:** We recommend installing dependencies inside a Miniforge-managed environment with **Python 3.11** and **NumPy version 1.25.2**.

To avoid versioning issues, use this command:

```bash
conda create -n nlp_env python=3.11 numpy=1.25.2
```

---

## ⚙️ Technical Workflow

1. **Text Preprocessing:**

   * Tokenization, lemmatization (spaCy)
   * Contractions expansion, stopword filtering

2. **Feature Engineering:**

   * Lexicon-based scoring using Loughran-McDonald dictionary
   * TF-IDF extraction for classical models
   * Embedding-based features for transformer input

3. **Models Applied:**

   * Logistic Regression, SVM, Random Forest
   * Multi-layer Perceptron (Keras)
   * FinBERT (Transformer)

4. **Evaluation Metrics:**

   * Accuracy, Precision, Recall, F1-Score
   * Confusion Matrix analysis

5. **Deployment Insight:**

   * FinBERT is best suited for real-world use
   * TF-IDF + MLP shows balance between accuracy and speed

---

## 📁 Project Structure

```
├── Dataset/               # Cleaned FPB, FiQA, and Annual Reports
├── Notebooks/            # Jupyter experiments
├── Wordlist/             # Custom lexicons used
├── README.md             # This file
```

---

## 🔧 Library Requirements

Full list of critical packages installed in the environment (`nlp_env`):

> **Python version**: 3.11.11
> **NumPy version**: 1.25.2

<details>
<summary>Click to expand key libraries</summary>

* `tensorflow==2.19.0`
* `keras==3.10.0`
* `transformers==4.52.4`
* `xgboost==1.7.6`
* `scikit-learn==1.3.2`
* `imbalanced-learn==0.12.4`
* `nltk==3.9.1`
* `spacy==3.8.7`
* `datasets==3.6.0`
* `evaluate==0.4.3`
* `huggingface-hub==0.32.4`
* `matplotlib==3.10.3`
* `seaborn==0.13.2`
* `optuna==4.3.0`

</details>

To install:

```bash
pip install -r requirements.txt
```

Or manually recreate from frozen conda list.

---

## 📄 Citation

If you use this project, please cite:

```
Aydas, A., Strandbygaard, L., Rettedal, M. (2025). Machine Learning Approaches for Financial Sentiment Analysis: Classifying Executive Letters in Danish Annual Reports. CBS.
```

---

Made with ❤ by Aydas, Strandbygaard & Rettedal

