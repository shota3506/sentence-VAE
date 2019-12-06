import os
import json
import torch
import argparse
import numpy as np

from model import VAE
from utils import decode_sentnece_from_token


def main(args):
    vocab = json.load(open(args.vocab_file, 'r') )

    w2i, i2w = vocab['w2i'], vocab['i2w']

    sos_idx = vocab['w2i']['<sos>']
    eos_idx = vocab['w2i']['<eos>']
    pad_idx = vocab['w2i']['<pad>']
    unk_idx = vocab['w2i']['<unk>']
    model = VAE(
        rnn_type=args.rnn_type,
        num_embeddings=len(w2i),
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
        max_sequence_length=args.max_sequence_length)

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s\n"%(args.load_checkpoint))

    # sample
    print('----------SAMPLES----------')
    model.eval()
    zs = torch.randn(args.num_samples, args.dim_latent)
    for i in range(len(zs)):
        z = zs[i].unsqueeze(0)
        output = model.infer(z)
        print(decode_sentnece_from_token(output[0].tolist(), i2w, eos_idx))
    print()

    print('-------INTERPOLATION-------')
    z1 = torch.randn([args.dim_latent]).numpy()
    z2 = torch.randn([args.dim_latent]).numpy()
    interpolation = np.zeros((args.dim_latent, args.num_samples))
    for dim, (s,e) in enumerate(zip(z1,z2)):
        interpolation[dim] = np.linspace(s, e, args.num_samples)
    interpolation = interpolation.T

    zs = torch.from_numpy(interpolation).float()
    for i in range(len(zs)):
        z = zs[i].unsqueeze(0)
        output = model.infer(z)
        print(decode_sentnece_from_token(output[0].tolist(), i2w, eos_idx))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('--vocab_file', type=str, default='data/vocab.json')
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
