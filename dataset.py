import os
import json
import torch
from torch.utils.data import Dataset


def collate_fn(data):
    data.sort(key=lambda x: x['length'], reverse=True)
    return torch.tensor([d['input'] for d in data]), torch.tensor([d['target'] for d in data]), \
        torch.tensor([d['length'] for d in data])


class SentenceDataset(Dataset):
    def __init__(self, data_file, vocab_file, max_sequence_length):
        super().__init__()
        self.data = []
        with open(data_file, 'r') as f:
            for line in f:
                self.data.append(line.strip().lower().split())

        self.vocab = json.load(open(vocab_file, 'r'))
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]

        input = ['<sos>'] + tokens
        input = input[:self.max_sequence_length]
        target = tokens[:self.max_sequence_length-1]
        target = target + ['<eos>']

        length = len(input)

        input.extend(['<pad>'] * (self.max_sequence_length - length))
        target.extend(['<pad>'] * (self.max_sequence_length - length))

        w2i = self.vocab['w2i']
        input = [w2i.get(token, w2i['<unk>'])for token in input]
        target = [w2i.get(token, w2i['<unk>'])for token in target]
      
        return {'input': input, 'target': target, 'length': length}

    @property
    def vocab_size(self):
        return len(self.vocab['i2w'])
