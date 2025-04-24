import torch

print(torch.cuda.is_available())  # Checking GPU availabilty on local system

import pandas as pd  # Importing required packages
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import BartForConditionalGeneration, TrainingArguments, Trainer,BartTokenizer, DataCollatorForSeq2Seq

df = pd.read_csv("filtered_mediasum.csv")  # Loading Mediasum dataset

# Quick preview
print("Loaded rows:", len(df))
print(df.columns)



# Applying Text Cleaning
df["document"] = df["document"].astype(str).apply(clean_text)
df["summary"] = df["summary"].astype(str).apply(clean_text)


# Randomly sampling 10000 rows from the dataset
df_sample = df


# Importing required packages


# Ensuring consistent datatypes
df_sample = df_sample[["document", "summary"]].dropna()
df_sample["document"] = df_sample["document"].astype(str)
df_sample["summary"] = df_sample["summary"].astype(str)

# Split into train and validation
train_df, val_df = train_test_split(df_sample, test_size=0.1, random_state=42)

# Convert to HuggingFace Datasets format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Importing required packages



# === 3. Load Tokenizer and Model ===
model_checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)

# === 4. Tokenization Function ===
def tokenize_function(example):
    model_inputs = tokenizer(
        example["document"]
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["summary"]
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# === 5. Training Config ===
training_args = TrainingArguments(
    output_dir="./distilbart_mediasum",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# === 6. Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# === 7. Train! ===
trainer.train()
