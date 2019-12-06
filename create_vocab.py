import os
import json
import argparse
from collections import defaultdict, Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def create_vocab(data, min_occ=1):
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    counter = OrderedCounter()
    w2i = {}
    i2w = []
    for d in data:
        counter.update(d)            

    for token in special_tokens:
        idx = len(w2i)
        i2w.append(token)
        w2i[token] = idx
    for token, c in counter.items():
        if c > min_occ and token not in special_tokens:
            idx = len(w2i)
            i2w.append(token)
            w2i[token] = idx
    
    assert len(w2i) == len(i2w)
    vocab = dict(w2i=w2i, i2w=i2w, min_occ=min_occ)
    return vocab


def main(args):
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append(line.strip().lower().split())

    print("Creating vocab...")
    vocab = create_vocab(data, min_occ=args.min_occ)
    print("Vocab size: %d" % len(vocab['w2i']))
    
    print("Dumping data...")
    json.dump(vocab, open(os.path.join(args.save_dir, 'vocab.json'), 'w'))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', type=str, default='data/quora/train.txt')
    parser.add_argument('--save_dir', type=str, default='data/quora')
    parser.add_argument('--min_occ', type=int, default=1)

    args = parser.parse_args()

    main(args)