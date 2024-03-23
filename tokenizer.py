from typing import List


class Tokenizer:
    def __init__(
        self,
        vocab_file,
        pad_index=0,
        unk_index=3,
    ):
        self.pad_index = pad_index
        self.unk_index = unk_index

        with open(vocab_file, "r") as f:
            self._idx2token = [line.strip() for line in f]
            self._token2idx = {tkn: i for i, tkn in enumerate(self._idx2token)}

    def encode(
        self,
        input: str,
    ) -> List[int]:
        tokens = input.split()
        return [
            self._token2idx[tkn] if tkn in self._token2idx else self.unk_index
            for tkn in tokens
        ]

    def decode(
        self,
        input,
    ) -> str:
        tokens = [self._idx2token[idx] for idx in input]
        return " ".join(tokens)
