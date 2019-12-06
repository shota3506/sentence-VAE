import os
import json
import torch
import argparse
from tqdm import tqdm

from dataset import SentenceDataset, collate_fn
from model import VAE
from loss import MaskedCrossEntropyLoss, KLLoss
from utils import decode_sentnece_from_token

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def main(args):
    dataset = SentenceDataset(args.data_file, args.vocab_file, args.max_sequence_length)

    sos_idx = dataset.vocab['w2i']['<sos>']
    eos_idx = dataset.vocab['w2i']['<eos>']
    pad_idx = dataset.vocab['w2i']['<pad>']
    unk_idx = dataset.vocab['w2i']['<unk>']
    model = VAE(
        rnn_type=args.rnn_type,
        num_embeddings=dataset.vocab_size,
        dim_embedding=args.dim_embedding,
        dim_hidden=args.dim_hidden, 
        num_layers=args.num_layers,
        bidirectional=args.bidirectional, 
        dim_latent=args.dim_latent, 
        word_dropout=args.word_dropout,
        dropout=args.dropout,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        pad_idx=pad_idx,
        unk_idx=unk_idx,
        max_sequence_length=args.max_sequence_length).to(device)

    print(model)
    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    hypotheses = []
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        pbar = dataset if args.print else tqdm(dataset)
        for d in pbar:
            input = torch.tensor([d['input']]).to(device)
            target = torch.tensor([d['target']]).to(device)
            length = torch.tensor([d['length']]).to(device)

            mean, logvar = model.encode(input, length)
            z = model.reparameterize(mean, logvar)
            ys = model.infer(z)

            original = decode_sentnece_from_token(target[0].tolist(), dataset.vocab['i2w'], eos_idx)
            hypothesis = decode_sentnece_from_token(ys[0].tolist(), dataset.vocab['i2w'], eos_idx)

            hypotheses.append(hypothesis)

            if args.print:
                print("-" * 100 + "\n")
                print("Original:   %s\nHypothesis: %s\n"  % (original, hypothesis))
    
    print("Saving...")

    with open(args.output_file, 'w') as f:
        f.write("\n".join(hypotheses))

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-p', '--print', action='store_true')

    parser.add_argument('--data_file', type=str, default='data')
    parser.add_argument('--vocab_file', type=str, default='data')
    parser.add_argument('--output_file', type=str, required=True)

    parser.add_argument('--max_sequence_length', type=int, default=50)

    # model settings
    parser.add_argument('-dl', '--dim_latent', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.6)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)

    # rnn settings
    parser.add_argument('-de', '--dim_embedding', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-dh', '--dim_hidden', type=int, default=256)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    main(args)
