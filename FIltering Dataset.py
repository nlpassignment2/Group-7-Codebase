from datasets import load_dataset
import pandas as pd

import requests

# Raw file URL (not the regular GitHub page link)
url = "https://raw.githubusercontent.com/zcgzcgzcg1/MediaSum/main/data/train_val_test_split.json"

# Download and save locally
response = requests.get(url)

if response.status_code == 200:
    with open("train_val_test_split.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("✅ File downloaded successfully.")
else:
    print(f"❌ Failed to download file. Status code: {response.status_code}")

dataset = pd.read_json("train_val_test_split.json")
dataset.to_csv("mediasum.csv", index=False)


# Function to count words
def word_count(text):
    return len(text)


# Filter records where the document has more than 1000 words
filtered_dataset = dataset.filter(
    lambda example: word_count(example["document"]) < 1500
)

# Convert to pandas DataFrame
df = pd.DataFrame(
    {"document": filtered_dataset["document"], "summary": filtered_dataset["summary"]}
)

# Save to CSV
df.to_csv("filtered_mediasum.csv", index=False)
print(f"Saved {len(df)} records to 'filtered_mediasum.csv'")
