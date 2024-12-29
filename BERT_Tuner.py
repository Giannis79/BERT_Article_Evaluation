import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import MSELoss
import json

# Custom Dataset Class
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']  # Access text from the DataFrame
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }

# Load Data
df = pd.read_csv('Trainer_Dataset.csv', encoding='latin-1') 

# Initialize Tokenizer and Model
model_name = 'sentiment_model_score'  # Load the pre-trained model
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1) 

# Data Loaders
max_len = 128
batch_size = 16

dataset = TextDataset(df, tokenizer, max_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Prediction Function
def predict_scores(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            predictions.extend(logits.tolist())

    return predictions

# Predict scores
predicted_scores = predict_scores(model, data_loader)

# Normalize scores to 0-10 range
min_score = min(predicted_scores)
max_score = max(predicted_scores)
normalized_scores = [(score - min_score) / (max_score - min_score) * 10 for score in predicted_scores] 

# Save normalization parameters
normalization_params = {'min_score': min_score, 'max_score': max_score}
with open('normalization_params.json', 'w') as f:
    json.dump(normalization_params, f)

# Create DataFrame with results
results_df = pd.DataFrame({'Text': df['text'], 'Predicted Score': normalized_scores})

# Save results to a CSV file
results_df.to_csv('predicted_scores_normalized.csv', index=False) 
