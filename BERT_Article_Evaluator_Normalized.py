import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Function to remove BOM from the first line
def remove_bom(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(filename, 'wb') as f:
        f.write(content)

# Load the pre-trained model and tokenizer
model_name = 'sentiment_model_score' 
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Load normalization parameters
with open(f"{model_name}/normalization_params.json", 'r') as f:
    normalization_params = json.load(f)
min_logit = normalization_params['min_score']
max_logit = normalization_params['max_score']

# Prediction Function
def predict_influence(model, tokenizer, text, min_logit, max_logit):
    if not isinstance(text, str):
        text = str(text)  # Ensure the text is a string
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    outputs = model(**inputs)
    logits = outputs.logits.item()
    # Normalize logits to (0, 1) range using min-max scaling
    normalized_score = (logits - min_logit) / (max_logit - min_logit)
    scaled_score = normalized_score * 9 + 1  # Scale to (1, 10) range
    return scaled_score

# Create the directory if it doesn't exist
output_dir = r'Kathimerini\reformatted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define file paths and months
files_and_months = [
    (r'Kathimerini\Kathimerini_articles_2022_02.csv', 'February'),
    (r'Kathimerini\Kathimerini_articles_2022_03.csv', 'March'),
    (r'Kathimerini\Kathimerini_articles_2022_04.csv', 'April'),
    (r'Kathimerini\Kathimerini_articles_2022_05.csv', 'May'),
    (r'Kathimerini\Kathimerini_articles_2022_06.csv', 'June'),
    (r'Kathimerini\Kathimerini_articles_2022_07.csv', 'July'),
    (r'Kathimerini\Kathimerini_articles_2022_08.csv', 'August')
]

# Process files and predict sentiments
results = []
for file_name, month in files_and_months:
    try:
        remove_bom(file_name)
        articles_df = pd.read_csv(file_name, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')
        for index, row in articles_df.iterrows():
            if 'Content' not in row:
                print(f"Error: 'Content' column is missing in {file_name}")
                continue
            content = row['Content']
            scaled_score = predict_influence(model, tokenizer, content, min_logit, max_logit)
            sentiment = 'Pro-Russian' if scaled_score < 5 else 'Pro-Ukrainian'
            results.append({
                'Sentiment': sentiment,
                'Score': scaled_score,
                'Month': month
            })
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Save results to a new CSV file
results_df.to_csv('Kathimerini_Evaluation.csv', index=False, encoding='utf-8-sig')

print("Evaluation completed and results saved to 'Kathimerini_Evaluation.csv'")
