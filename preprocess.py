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


def create_vocab(questions, idx, min_occ=1):
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    counter = OrderedCounter()
    w2i = {}
    i2w = []
    for i in idx:
        question = questions[i]
        tokens = [token.lower() for s in question['annotation'] for token in s['tokens']]
        counter.update(tokens)            

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


def preprocess(questions, vocab, max_sequence_length=50):
    w2i = vocab['w2i']

    idx = list(questions.keys())

    preprocessed_questions = {}
    for i in idx:
        question = questions[i]
        tokens = [token.lower() for s in question['annotation'] for token in s['tokens']]

        input = ['<sos>'] + tokens
        input = input[:max_sequence_length]
        target = tokens[:max_sequence_length-1]
        target = target + ['<eos>']
        length = len(input)

        assert len(input) == len(target)
        
        input.extend(['<pad>'] * (max_sequence_length - length))
        input = [w2i.get(token, w2i['<unk>'])for token in input]
        target.extend(['<pad>'] * (max_sequence_length - length))
        target = [w2i.get(token, w2i['<unk>'])for token in target]

        preprocessed_questions[i] = {
            'input': input,
            'target': target,
            'length': length
        }

    return preprocessed_questions

def main(args):
    questions = json.load(open(os.path.join(args.data_dir, 'annotated_questions.json'), 'r'))
    
    data = json.load(open(args.train_file, 'r'))
    idx = [d['original'] for d in data]

    print("Creating vocab...")
    vocab = create_vocab(questions, idx, min_occ=args.min_occ)
    print("Vocab size: %d" % len(vocab['w2i']))

    print("Preprocessing data...")
    preprocessed_questions = preprocess(questions, vocab, max_sequence_length=args.max_sequence_length)
    
    print("Dumping data...")
    json.dump(vocab, open(os.path.join(args.data_dir, 'vocab.json'), 'w'))
    json.dump(preprocessed_questions, open(os.path.join(args.data_dir, 'preprocessed_questions.json'), 'w'))
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data/quora')
    parser.add_argument('--train_file', type=str, default='data/quora/train.json')
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--max_sequence_length', type=int, default=50)

    args = parser.parse_args()

    main(args)