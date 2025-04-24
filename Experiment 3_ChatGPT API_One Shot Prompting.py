import pandas as pd
import random
from rouge import Rouge
import requests
import time


# Github Model API only allows 50 requests total
# Function to summarize a document using GPT-4
def summarize_document(document):
    # API URL for the GPT-4.1 model completion
    token = "<add your token here>"  # Your API token
    endpoint = "https://models.github.ai/inference"  # The endpoint for the API
    model = "openai/gpt-4.1"  # The model name
    api_url = f"{endpoint}/chat/completions"
    # Constructing the body of the request similar to the JS example
    body = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional interview summariser.",
            },
            {
                "role": "user",
                "content": "The following is the given interview transcript \n"
                + "STEVE INSKEEP, HOST: Good morning. I'm Steve Inskeep. This happens to many people who get a chance to be on TV. Michelle Obama turned up on the Grammys telecast. And afterwards, she says she received a text message from her mom, who said the former first lady had failed to tell her she'd be on TV. Mrs. Obama said she thought she had, but her mom was having none of that. In a deft bit of motherly guilting, Marian Robinson wrote her daughter, I saw it because someone else called me."
                + "\n and its summary \n"
                + "Former first lady Michelle Obama was on the Grammy Awards over the weekend. Marian Robinson texted her daughter: "
                "I saw it, because (someone else) called me.\n"
                + "understand how the summary was made and key theme was identified now i am giving you another document create its summary the output should be just one paragraph and nothing else \n"
                + document,
            },
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "model": model,
    }

    # Set the headers for authentication and content type
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    # Make the POST request to the API
    response = requests.post(api_url, json=body, headers=headers)

    if response.status_code == 200:
        # Extract the generated summary from the response
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


# Load your CSV file
df = pd.read_csv("filtered_mediasum.csv")  # The CSV file with 'document' and 'summary'

# Sample 100 random rows
sampled_df = df.sample(n=50, random_state=42)  # Randomly select 100 rows

# Prepare to store results
gpt_summaries = []

# Loop through each document and generate summaries
num_requests = 0
for index, row in sampled_df.iterrows():
    document = row["document"]
    summary = row["summary"]

    # Get the GPT-4 summary
    gpt_summary = summarize_document(document)
    print(gpt_summary)
    num_requests += 1
    if num_requests % 8 == 0:
        print("Waiting")
        time.sleep(60)
        print("WaitOver")

    # Append results
    gpt_summaries.append([document, summary, gpt_summary])

# Create a new DataFrame for the results
results_df = pd.DataFrame(gpt_summaries, columns=["document", "summary", "GPT Summary"])

# Save the new DataFrame to a CSV
results_df.to_csv("output_file.csv", index=False)

# Now calculate NLP evaluation metrics (e.g., ROUGE scores)
rouge = Rouge()

# Initialize score variables
rouge_scores = []

# Calculate ROUGE scores for each document-summary pair
for _, row in results_df.iterrows():
    original_summary = row["summary"]
    gpt_summary = row["GPT Summary"]

    scores = rouge.get_scores(gpt_summary, original_summary)
    rouge_scores.append(scores[0])  # Collect ROUGE scores for each pair

# Compute average ROUGE scores (for ROUGE-1, ROUGE-2, ROUGE-L)
avg_rouge_1 = sum([score["rouge-1"]["f"] for score in rouge_scores]) / len(rouge_scores)
avg_rouge_2 = sum([score["rouge-2"]["f"] for score in rouge_scores]) / len(rouge_scores)
avg_rouge_L = sum([score["rouge-l"]["f"] for score in rouge_scores]) / len(rouge_scores)

# Print the average ROUGE scores
print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
print(f"Average ROUGE-L: {avg_rouge_L:.4f}")
