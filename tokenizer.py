class Tokenizer:
    def __init__(
        self,
        vocab_file,
        pad_index=0,
        bos_index=1,
        eos_index=2,
        unk_index=3,
    ):
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.bos_index = bos_index
        self.eos_index = eos_index

        with open(vocab_file, "r") as f:
            self._idx2token = [line.strip() for line in f]
            self._token2idx = {tkn: i for i, tkn in enumerate(self._idx2token)}

    def __len__(self) -> int:
        return len(self._idx2token)

    def encode(
        self,
        input: str,
    ):
        tokens = input.split()
        return [
            self._token2idx[tkn] if tkn in self._token2idx else self.unk_index
            for tkn in tokens
        ]

    def decode(
        self,
        input,
    ):
        tokens = [self._idx2token[idx] for idx in input]
        return " ".join(tokens)
