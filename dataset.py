import os
import json
import torch
from torch.utils.data import Dataset


def collate_fn(data):
    data.sort(key=lambda x: x['length'], reverse=True)
    return torch.tensor([d['input'] for d in data]), torch.tensor([d['target'] for d in data]), \
        torch.tensor([d['length'] for d in data])


class QuoraDataset(Dataset):
    def __init__(self, data_file, question_file, vocab_file):
        super().__init__()
        self.data = json.load(open(data_file, 'r'))
        self.idx = [d['original'] for d in self.data]

        self.questions = json.load(open(question_file, 'r'))
        self.vocab = json.load(open(vocab_file, 'r'))

        self.questions = {i: self.questions[i] for i in self.idx + [d['reference'] for d in self.data]}

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        idx = self.idx[i] 
        question = self.questions[idx]
        return {'input': question['input'], 'target': question['target'], 'length': question['length']}

    @property
    def vocab_size(self):
        return len(self.vocab['i2w'])
