import os
import gc
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from evaluate import load as load_metric

# 1. Load the MediaSum dataset
dataset = load_dataset("ccdv/mediasum")

# 2. Rename columns for clarity
dataset = dataset.rename_columns({"document": "input_text", "summary": "target_text"})

# 3. Tokenizer and preprocessing
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


def preprocess_function(examples):
    model_inputs = tokenizer(examples["input_text"])
    labels = tokenizer(
        examples["target_text"]
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Tokenize
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    
)

# 4. Load model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# 5. Training config
training_args = TrainingArguments(
    output_dir="./mediasum_bart",
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # adjust based on VRAM
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

# 6. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 8. Train
trainer.train()

trainer.save_model("./distilbart-mediasum-final")
tokenizer.save_pretrained("./distilbart-mediasum-final")

# #Evaluating

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
val_df = tokenized_dataset["test"].to_pandas()
val_sample = val_df.sample(n=500, random_state=42).copy()
val_sample["generated_summary"] = val_sample["document"].apply(generate_summary)

val_sample_out = val_sample[["document", "summary", "generated_summary"]]
val_sample_out.to_csv("distilbart_mediasum_generated_summaries.csv", index=False)



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
