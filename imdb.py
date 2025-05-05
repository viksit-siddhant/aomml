import torch
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # Example tokenizer

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)



def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

def text_pipeline(text):
    tokens = tokenizer(text)
    numericalized_tokens = vocab(tokens)

    if len(numericalized_tokens) > 256:
        processed_tokens = numericalized_tokens[:256]
    else:
        padding_needed = 256 - len(numericalized_tokens)
        processed_tokens = numericalized_tokens + [PAD_IDX] * padding_needed

    return torch.tensor(processed_tokens, dtype=torch.long)

def collate_batch(batch):
    processed_texts = []
    labels = []
    for text, label in batch:
        processed_text = text_pipeline(text)
        processed_texts.append(processed_text)
        labels.append(torch.tensor(label, dtype=torch.long))

    texts_tensor = torch.stack(processed_texts, dim=0)
    labels_tensor = torch.stack(labels, dim=0)

    return texts_tensor, labels_tensor




imdb = datasets.load_dataset('imdb', split='train')
tokenized_imdb = imdb.map(preprocess_function, batched=True)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        #return len(self.data)
        return 500

    def __getitem__(self, idx):
        idx = idx*12395234 + 183349213
        idx = idx % len(self.data)
        item = self.data[idx]
        text = item['input_ids']
        label = item['label']
        return torch.tensor(text), torch.tensor(label)
    
dataset = Dataset(tokenized_imdb)
train_size = int(0.99 * len(dataset))
val_size = len(dataset) - train_size
train,val = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(
    train,
    batch_size=1,
    shuffle=True,
)
val_dataloader = DataLoader(
    val,
    batch_size=32,
    shuffle=True,
)

# Define the vocabulary size and padding index
vocab_size = len(tokenizer)
PAD_IDX = tokenizer.pad_token_id