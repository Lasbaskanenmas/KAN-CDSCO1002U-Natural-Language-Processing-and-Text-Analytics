import os
import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


import spacy
import re
import contractions
import unicodedata
import nltk
import string

# Load nltk stopwords
from nltk.corpus import stopwords


import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import warnings


# FinBERT


#!pip install -q scikit-learn datasets torch transformers evaluate optuna


#!pip install "numpy<2.0.0"


# Step 1: Import Libraries

import torch
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import evaluate


# Step 2: Load the dataset
df = pd.read_csv("Dataset/Cleaned Sentence Data/Cleaned Annual Reports.csv")  # Replace with your dataset path
assert 'Sentence' in df.columns and 'Sentiment' in df.columns, "CSV must have 'Sentence' and 'Sentiment' columns"

# Step 3: Encode labels if needed
if df['Sentiment'].dtype == object:
    le = LabelEncoder()
    df['Sentiment'] = le.fit_transform(df['Sentiment'])  # Save le.classes_ if needed for decoding

# Step 4: Convert to Hugging Face Dataset and split
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.3, seed=42)
temp = dataset["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    'train': dataset['train'],
    'validation': temp['train'],
    'test': temp['test']
})

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Step 5: Tokenization
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['Sentence'], truncation=True, padding='max_length', max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Remove non-numeric columns
for d in [train_dataset, val_dataset, test_dataset]:
    if '__index_level_0__' in d.column_names:
        d = d.remove_columns(['Sentence', '__index_level_0__'])
    else:
        d = d.remove_columns(['Sentence'])


train_dataset = train_dataset.rename_column("Sentiment", "labels")
val_dataset = val_dataset.rename_column("Sentiment", "labels")
test_dataset = test_dataset.rename_column("Sentiment", "labels")

train_dataset = train_dataset.with_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
val_dataset = val_dataset.with_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
test_dataset = test_dataset.with_format("torch", columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])


# Step 6: Load FinBERT model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(df['Sentiment'].unique())
)


# Load F1 metric (can also add accuracy if desired)
f1_metric = evaluate.load("f1")
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"]
    }

def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(df['Sentiment'].unique())
    )

def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
    }

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="Natural Language Processing/Models/optimized_finbert_sentiment",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# Step 8: Initialize Trainer with early stopping
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # stops after 5 epochs with no improvement
)

best_run = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    n_trials=10,  # increase for more thorough search
    hp_space=hp_space,
    compute_objective=lambda metrics: metrics["eval_f1"]
)

best_args = training_args

# Update args with best values
best_args.learning_rate = best_run.hyperparameters["learning_rate"]
best_args.per_device_train_batch_size = best_run.hyperparameters["per_device_train_batch_size"]
best_args.weight_decay = best_run.hyperparameters["weight_decay"]
best_args.num_train_epochs = best_run.hyperparameters["num_train_epochs"]

# Recreate Trainer with best config
trainer = Trainer(
    model_init=model_init,
    args=best_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Step 9: Train
trainer.train()

# Step 10: Save the best model
#trainer.save_model("Natural Language Processing/Models/final_finbert_model")
history = trainer.state.log_history
history_df = pd.DataFrame(history)
#history_df.to_csv("Natural Language Processing/Models/training_history.csv", index=False)

#tokenizer.save_pretrained("Natural Language Processing/Models/final_finbert_model")

# Step 11: Evaluate
preds_output = trainer.predict(test_dataset)
pred_labels = np.argmax(preds_output.predictions, axis=1)
true_labels = preds_output.label_ids

report = classification_report(true_labels, pred_labels, target_names=le.classes_, digits=4)
print(report)


# Compute confusion matrix
cm = confusion_matrix(true_labels, pred_labels)

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()


print(report)


# Convert training history to DataFrame
# Plot all in one figure

epochs_list = [1, 2, 3, 4, 5, 6, 7]

train_history = history_df[history_df["epoch"].isin(epochs_list)]
# Only drop index if epoch==5 exists
epoch5_idx = train_history[train_history["epoch"] == 5].index
if len(epoch5_idx) > 0:
	train_history = train_history.drop(epoch5_idx[0]).reset_index(drop=True)

plt.figure(figsize=(10, 6))

# Accuracy
#plt.plot(train_history['eval_f1'], label='Val F1 Score')
plt.plot(train_history['eval_accuracy'], label='Val Accuracy')

# Loss
#plt.plot(history_df['loss'], label='Train Loss')
plt.plot(train_history['eval_loss'], label='Val Loss')

plt.title('Training and Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
