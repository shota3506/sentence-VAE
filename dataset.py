import os
import io
import json
import torch
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer


def collate_fn(data):
    data.sort(key=lambda x: x['length'], reverse=True)
    return torch.stack([d['input'] for d in data]), torch.stack([d['target'] for d in data]), torch.tensor([d['length'] for d in data])


class ParaphraseDataset(Dataset):
    def __init__(self, data_dir, split, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        
        vocab_file = os.path.join(data_dir, "vocab.json")
        data_file = os.path.join(data_dir, "%s.json" % split)

        assert os.path.exists(vocab_file)
        assert os.path.exists(data_file)

        self._load_vocab(vocab_file)
        self._load_data(data_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.data[idx]['input']), 
            'target': torch.tensor(self.data[idx]['target']), 
            'length': self.data[idx]['length']}

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def _load_vocab(self, vocab_file):
        vocab = json.load(open(vocab_file, 'r'))
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_data(self, data_file):
        self.data = json.load(open(data_file, 'r'))


