import torch

print(torch.cuda.is_available())  # Checking GPU availabilty on local system

import pandas as pd  # Importing required packages
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

df = pd.read_csv("filtered_mediasum.csv")  # Loading Mediasum dataset


# Quick preview
print("Loaded rows:", len(df))
print(df.columns)


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Applying Text Cleaning
df["document"] = df["document"].astype(str).apply(clean_text)
df["summary"] = df["summary"].astype(str).apply(clean_text)

# Importing required packages
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Ensuring consistent datatypes
df_sample = df[["document", "summary"]].dropna()
df_sample["document"] = df_sample["document"].astype(str)
df_sample["summary"] = df_sample["summary"].astype(str)

# Split into train and validation
train_df, val_df = train_test_split(df_sample, test_size=0.1, random_state=42)

# Convert to HuggingFace Datasets format
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Importing required packages
from transformers import BartTokenizer


# Creating model variable for DistilBART
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = BartTokenizer.from_pretrained(model_name)


# Tokenization function
def tokenize_function(example):
    model_inputs = tokenizer(
        example["document"], max_length=512, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["summary"], max_length=128, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply tokenization
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)


# Defining model and storing it on GPU
from transformers import BartForConditionalGeneration, TrainingArguments, Trainer

model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = model.to(device)


# Defining training parameters
training_args = TrainingArguments(
    output_dir="./distilbart-mediasum",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,  # <-- Should work on latest versions
    logging_dir="./logs",
    save_total_limit=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    fp16=True,  # Optional, if you're using GPU with float16 support
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./distilbart-mediasum-final")
tokenizer.save_pretrained("./distilbart-mediasum-final")


# Evaluating Model on Validation Set

model_path = "./distilbart-mediasum-final"

tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")


def generate_summary(text):
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    ).to(model.device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"], max_length=512, num_beams=4, early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# Run on a small validation subset
val_sample = val_df.sample(n=500, random_state=42).copy()
val_sample["generated_summary"] = val_sample["document"].apply(generate_summary)

val_sample_out = val_sample[["document", "summary", "generated_summary"]]
val_sample_out.to_csv("output.csv", index=False)

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def compute_rouge_scores(reference, generated):
    score = scorer.score(reference, generated)
    return {
        "rouge1": score["rouge1"].fmeasure,
        "rouge2": score["rouge2"].fmeasure,
        "rougeL": score["rougeL"].fmeasure,
    }


rouge_results = val_sample.apply(
    lambda row: compute_rouge_scores(row["summary"], row["generated_summary"]), axis=1
)
rouge_df = pd.DataFrame(list(rouge_results))

# Print average ROUGE scores
print("Average ROUGE scores:")
print(rouge_df.mean())
