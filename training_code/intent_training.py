import os
import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import Trainer,TrainingArguments
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

df = pd.read_csv("./intent_dataset/intent_dataset.csv")

# split dataset
train_valid_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Intent"])
train_df, valid_df = train_test_split(train_valid_df, test_size=0.1, random_state=42, stratify=train_valid_df["Intent"])
le = LabelEncoder()
le.fit(df["Intent"])
train_df["Intent"] = le.transform(train_df["Intent"])
valid_df["Intent"] = le.transform(valid_df["Intent"])
test_df["Intent"] = le.transform(test_df["Intent"])

# convert to huggingface dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)
test_dataset = Dataset.from_pandas(test_df)

# initialize tokenizer
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['Query'], padding=True, truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
valid_dataset = valid_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Rename label column to 'labels'
train_dataset = train_dataset.rename_column("Intent", "labels")
valid_dataset = valid_dataset.rename_column("Intent", "labels")
test_dataset = test_dataset.rename_column("Intent", "labels")

# Set dataset format for PyTorch
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
valid_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# initialize the model
num_labels = len(le.classes_)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# training parameters
training_args = TrainingArguments(output_dir="./bert_checkpoints",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"
    )

# evaluation metrics
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

# print the checkpoints
ckpt_root = "./bert_checkpoints"

# Just list the checkpoint directories as-is
all_ckpts = [d for d in os.listdir(ckpt_root) if d.startswith("checkpoint")]
print("Epoch-wise checkpoints:")
for idx, ckpt in enumerate(all_ckpts, start=1):
    print(f"Epoch {idx}")
    print(f"  {ckpt}")

# evaluate the trained model
metrics = trainer.evaluate(test_dataset)
print(metrics)
print("Best checkpoint:", trainer.state.best_model_checkpoint)

# print total number of training data, validation data, test data
print("Total number of training data: ", len(train_df))
print("Total number of validation data: ", len(valid_df))
print("Total number of test data: ", len(test_df))

# print total number of banking and non banking data in training set
print("Total number of banking data in training set", len(train_df[train_df['Intent']==0]))
print("Total number of non banking in training set", len(train_df[train_df['Intent']==1]))

# Get predictions
predictions_output = trainer.predict(test_dataset)

# Predicted class labels
y_pred = np.argmax(predictions_output.predictions, axis=1)

# True labels
y_true = predictions_output.label_ids

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)
MODEL_DIR = './intent_model'

# Save model & tokenizer
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# Save label encoder
joblib.dump(le, f'{MODEL_DIR}/label_encoder.pkl')
print(os.listdir(MODEL_DIR))