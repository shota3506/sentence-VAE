import os
import json
import argparse
from nltk.tokenize import TweetTokenizer
from collections import defaultdict, Counter, OrderedDict
from tqdm import tqdm

tokenizer = TweetTokenizer(preserve_case=False)


class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


def create_vocab(data_file, min_occ=1):
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

    counter = OrderedCounter()
    w2i = {}
    i2w = []

    for token in special_tokens:
        idx = len(w2i)
        i2w.append(token)
        w2i[token] = idx

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            words = tokenizer.tokenize(line)
            counter.update(words)

    for token, c in counter.items():
        if c > min_occ and token not in special_tokens:
            idx = len(w2i)
            i2w.append(token)
            w2i[token] = idx
    
    assert len(w2i) == len(i2w)
    vocab = dict(w2i=w2i, i2w=i2w)
    return vocab


def create_data(data_file, vocab, max_sequence_length):
    data = []
    w2i = vocab['w2i']
    with open(data_file, 'r') as f:
        pbar = tqdm(f)
        pbar.set_description('Preprocess %s' % data_file)
        for i, line in enumerate(pbar):
            words = tokenizer.tokenize(line)
            input = ['<sos>'] + words
            input = input[:max_sequence_length]

            target = words[:max_sequence_length-1]
            target = target + ['<eos>']

            assert len(input) == len(target), "%i, %i"%(len(input), len(target))
            length = len(input)

            input.extend(['<pad>'] * (max_sequence_length-length))
            target.extend(['<pad>'] * (max_sequence_length-length))

            input = [w2i.get(w, w2i['<unk>']) for w in input]
            target = [w2i.get(w, w2i['<unk>']) for w in target]

            id = len(data)
            data.append({'input': input, 'target': target, 'length': length})
    return data


def main(args):
    train_file = os.path.join(args.path, 'train.txt')
    valid_file = os.path.join(args.path, 'valid.txt')
    test_file = os.path.join(args.path, 'test.txt')

    print("Creating vocab...")
    vocab = create_vocab(train_file, min_occ=args.min_occ)
    print("Vocab size: %d" % len(vocab['i2w']))
    json.dump(vocab, open(os.path.join(args.path, 'vocab.json'), 'w'))

    train_data = create_data(train_file, vocab, args.max_sequence_length)
    valid_data = create_data(valid_file, vocab, args.max_sequence_length)
    test_data = create_data(test_file, vocab, args.max_sequence_length)

    json.dump(train_data, open(os.path.join(args.path, 'train.json'), 'w'))
    json.dump(valid_data, open(os.path.join(args.path, 'valid.json'), 'w'))
    json.dump(test_data, open(os.path.join(args.path, 'test.json'), 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data/ptb')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)

    args = parser.parse_args()

    main(args)
