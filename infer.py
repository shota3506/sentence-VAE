import os
import json
import torch
import argparse
import numpy as np

from models.vae import VAE
from utils import decode_sentnece_from_token


def main(args):
    vocab_file = os.path.join(args.data_dir, 'vocab.json')
    vocab = json.load(open(vocab_file, 'r') )

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = VAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional)

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s\n"%(args.load_checkpoint))


    # sample
    print('----------SAMPLES----------')
    model.eval()
    z = torch.randn(args.num_samples, args.latent_size)
    output = model.infer(z)
    for i in range(len(output)):
        print(decode_sentnece_from_token(output[i].tolist(), i2w, w2i['<eos>']))
    print()

    print('-------INTERPOLATION-------')
    z1 = torch.randn([args.latent_size]).numpy()
    z2 = torch.randn([args.latent_size]).numpy()
    interpolation = np.zeros((args.latent_size, args.num_samples))
    for dim, (s,e) in enumerate(zip(z1,z2)):
        interpolation[dim] = np.linspace(s, e, args.num_samples)
    interpolation = interpolation.T

    z = torch.from_numpy(interpolation).float()
    output = model.infer(z)
    for i in range(len(output)):
        print(decode_sentnece_from_token(output[i].tolist(), i2w, w2i['<eos>']))

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
