import torch
import torch.nn as nn
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader


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




train_dataset = IMDB(root='.', split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(
    yield_tokens(train_dataset),
    min_freq=20,
    specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])
PAD_IDX = vocab['<pad>']

train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_batch
)