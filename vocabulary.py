from typing import Dict, Optional, List


class Vocabulary:
    def __init__(self, tokens: List[str]):
        self._tokens = tokens
        self._index = {token: idx for idx, token in enumerate(tokens)}

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, token: str) -> int:
        if token in self._index:
            return self._index[token]
        return -1

    def __contains__(self, token: str) -> bool:
        return token in self._index

    def tokens(self) -> List[str]:
        return self._tokens

    def lookup(self, idx: int) -> str:
        return self._tokens[idx]


def build_vocabulary(
    ordered_dict: Dict, max_length: int, specials: Optional[List[str]] = None
) -> Vocabulary:
    specials = specials or []
    for token in specials:
        ordered_dict.pop(token, None)

    length = len(specials)
    tokens = []
    for token, freq in ordered_dict.items():
        tokens.append(token)
        length += 1
        if length >= max_length:
            break

    tokens[0:0] = specials
    return Vocabulary(tokens)
