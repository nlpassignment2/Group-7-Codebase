import pandas as pd #Importing required packages
import re
import string
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


#Loading Data
df = pd.read_json(r"C:\Users\ccqer\Downloads\news_dialogue\news_dialogue.json")

# Quick preview
print("Loaded rows:", len(df))
print(df.columns)


def clean_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)


#Cleaning Data
df['utt'] = df['utt'].astype(str).apply(clean_text)
df['summary'] = df['summary'].astype(str).apply(clean_text)

def compute_lcs(a, b):
    """Compute the length of the Longest Common Subsequence between two strings."""
    matcher = SequenceMatcher(None, a, b)
    return sum(triple.size for triple in matcher.get_matching_blocks())


#Random Sampling
df_sample = df.sample(n=10000, random_state=42)

from difflib import SequenceMatcher

# Compute LCS for each row
df_sample['lcs_len'] = df_sample.apply(lambda row: compute_lcs(row['utt'], row['summary']), axis=1)

# Normalizing by summary or dialogue length
df_sample['lcs_ratio_summary'] = df_sample['lcs_len'] / df_sample['summary'].str.len()
df_sample['lcs_ratio_utt'] = df_sample['lcs_len'] / df_sample['utt'].str.len()

print(df_sample[['summary', 'lcs_len', 'lcs_ratio_summary']].head())

import seaborn as sns
import matplotlib.pyplot as plt

#style for prettier visuals
sns.set(style="whitegrid")

# Plotting histogram of LCS ratio
plt.figure(figsize=(10, 6))
sns.histplot(df_sample['lcs_ratio_summary'], bins=50, kde=True, color='skyblue')

plt.title('Extractiveness of MediaSum Summaries (LCS Ratio)', fontsize=16)
plt.xlabel('LCS Ratio (summary)', fontsize=14)
plt.ylabel('Number of Examples', fontsize=14)
plt.axvline(x=0.8, color='red', linestyle='--', label='80% Extractive')
plt.legend()
plt.tight_layout()
plt.show()


#Defining calculation of summary repetitiion score
def repetition_score(text, n=2):
    tokens = text.split()
    ngrams = list(zip(*[tokens[i:] for i in range(n)]))
    return 1 - len(set(ngrams)) / max(1, len(ngrams))

df['summary_repetition'] = df['summary'].apply(repetition_score) 

print(df['summary_repetition'])


#Plotting summary repetition score
sns.histplot(df['summary_repetition'], bins=20, kde=False)
plt.title('Summary Repetition (Bigram Level)')
plt.xlabel('Repetition Score')
plt.ylabel('Number of Samples')
plt.show()