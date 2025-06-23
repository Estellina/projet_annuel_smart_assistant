import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, EncoderDecoderModel

class QuizDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        target = self.tokenizer(
            item['question'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': target['input_ids'].squeeze()
        }

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

# Generation function
def generate_qcm(model, tokenizer, text, device='cpu'):
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        generated = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=256
        )
    return tokenizer.decode(generated[0], skip_special_tokens=True)
