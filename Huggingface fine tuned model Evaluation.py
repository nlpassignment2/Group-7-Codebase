import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import evaluate  # NEW way to load metrics

# Optional: for BERTScore
import bert_score

# Load model and tokenizer
model_name = "ccdv/lsg-bart-base-4096-mediasum"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

import torch
import evaluate
import pandas as pd
import math

df = pd.read_csv("filtered_mediasum.csv")  # Replace with your actual CSV
df = df.dropna(subset=["document"])  # , 'summary'])  # Remove rows with missing values
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

# Metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

# Generate summaries
generated_summaries = []
print(
    "Generating summaries on CUDA..."
    if device.type == "cuda"
    else "Generating on CPU..."
)

for doc in tqdm(df["document"].tolist()):
    inputs = tokenizer(doc, return_tensors="pt").to(device)
    # Dynamic summary length using log scale
    a = 20  # scale factor
    b = 30  # base length
    summary_len = int(a * math.log(inputs["input_ids"].shape[1]) + b)

    # Optionally add min/max bounds
    min_output_len = max(30, int(summary_len * 0.4))
    max_output_len = int(summary_len)
    summary_ids = model.generate(
        inputs["input_ids"],
        min_length=min_output_len,
        max_length=max_output_len,
        num_beams=4,
        length_penalty=1.2,
    )
    generated = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    generated_summaries.append(generated)

df["generated_summary"] = generated_summaries
references = df["summary"].tolist()

# Evaluate: ROUGE
rouge_results = rouge.compute(
    predictions=generated_summaries, references=references, use_stemmer=True
)
print("\nROUGE scores:")
for key in rouge_results:
    print(f"{key}: {rouge_results[key]:.4f}")

# Evaluate: BLEU
bleu_results = bleu.compute(
    predictions=generated_summaries, references=[[ref] for ref in references]
)
print(f"\nBLEU score: {bleu_results['score']:.4f}")

# Evaluate: BERTScore
print("\nCalculating BERTScore (on GPU if available)...")
P, R, F1 = bert_score.score(generated_summaries, references, lang="en", device=device)
print(
    f"BERTScore:\nPrecision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}"
)

# # Save output
df.to_csv("output_with_generated_summaries.csv", index=False)
print("\nSaved: output_with_generated_summaries.csv")
