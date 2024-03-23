import torch
from torch.utils.data import Dataset


class SentenceDataset(Dataset):
    def __init__(
        self,
        data_file,
        tokenizer,
        max_length=50,
        bos_index=1,
        eos_index=2,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.bos_index = bos_index
        self.eos_index = eos_index

        with open(data_file, "r") as f:
            self._data = [line.strip() for line in f]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        d = self._data[idx]
        d = (
            [self.bos_index]
            + self.tokenizer(d)[: self.max_length - 2]
            + [self.eos_index]
        )
        return d

    @staticmethod
    def collate_fn(data):
        s = [torch.tensor(d) for d in data]
        s = torch.nn.utils.rnn.pad_sequence(s, batch_first=True)
        return s
