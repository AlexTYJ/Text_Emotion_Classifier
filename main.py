import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
from collections import namedtuple
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import torch.nn.functional as F

import module

def train(model, config, train_dataset, eval_dataset):
    CE = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = config.lr)
    model.train()
    global_step = 0
    best_acc = 0
    for epoch in range(config.num_epoch):
        for data in train_dataset:
            logits = model(data[0])
            loss = CE(logits, data[1])
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            if global_step % config.eval_interval == 0:
                print('epoch: {}, global_step: {}, loss: {:.4f}'.format(epoch, global_step, loss.item()))
                acc = evaluate(model, eval_dataset)
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), config.save_path)
    return

def evaluate(model, dataset):
    model.eval()
    pred = []
    golden = []
    with torch.no_grad():
        for data in dataset:
            pred.extend(model.predict(data[0]).reshape(1, -1).squeeze().int().tolist())
            golden.extend(data[1].reshape(1, -1).squeeze().int().tolist())
    model.train()
    acc = cal_acc(pred, golden)
    print('acc: {:.4f}'.format(acc))
    return acc

def cal_acc(pred, golden):
    pred = np.array(pred)
    golden = np.array(golden)
    correct = np.sum(pred == golden)
    total = len(golden)
    return correct / total

class SentimentDataset(Dataset):
    def __init__(self, data_path, vocab_path) -> None:
        self.vocab = json.load(open(vocab_path, 'r', encoding='utf-8'))
        self.data = self.load_data(data_path)
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f]
            random.shuffle(raw_data)
        data = []
        for item in raw_data:
            text = item['text']
            text_id = [self.vocab[t] if t in self.vocab.keys() else self.vocab['UNK'] for t in text]
            label = int(item['label'])
            data.append([text_id, label])
        return data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

def collate_fn(data):
    pad_idx = 8019
    texts = [d[0] for d in data]
    label = [d[1] for d in data]
    batch_size = len(texts)
    max_length = max([len(t) for t in texts])
    text_ids = torch.ones((batch_size, max_length)).long().fill_(pad_idx)
    label_ids = torch.tensor(label).long()
    for idx, text in enumerate(texts):
        text_ids[idx, :len(text)] = torch.tensor(text)
    return text_ids, label_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default='save_model/best.pt')
    parser.add_argument('--train', default='data/train.jsonl')
    parser.add_argument('--test', default='data/test.jsonl')
    parser.add_argument('--val', default='data/val.jsonl')
    parser.add_argument('--num_epoch', default=25, type=int)
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--vocab', default='data/vocab.json')
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--filter_sizes', default=[4,4,5,5,6,6], type=list)

    parser.add_argument('--model', default='TextTransformer')
    parser.add_argument('--window_size', default=512, type=int)
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    arg = parser.parse_args()

    train_dataset = SentimentDataset(arg.train, arg.vocab)
    val_dataset = SentimentDataset(arg.val, arg.vocab)
    test_dataset = SentimentDataset(arg.test, arg.vocab)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size,collate_fn=collate_fn,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=arg.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)
    chr_vocab = json.load(open(arg.vocab, 'r', encoding = 'utf-8'))
    config = {
        'dropout': arg.dropout,
        'num_classes': 2,
        'vocab_size': len(chr_vocab),
        'embedding_dim':arg.hidden_dim,
        'filter_sizes':arg.filter_sizes,
        'window_size':arg.window_size,
        'num_heads':arg.num_heads,
        'num_layers':arg.num_layers
    }
    config = namedtuple('config', config.keys())(**config)
    if arg.model == 'TextCNN':
        model = module.TextCNN(config)
    else:
        model = module.TextTransformer(config)
    
    train(model, arg, train_loader, val_loader)