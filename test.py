import os
import json
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from models.rnn_vae import RNNVAE
from models.transformer_vae import TransformerVAE
from loss import MaskedCrossEntropyLoss, KLLoss
from utils import load_dataset, decode_sentnece_from_token

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def main(args):
    dataset, loader = load_dataset(args.data_dir, 'test', 1)

    if args.model == 'rnn':
        model = RNNVAE(
            rnn_type=args.rnn_type,
            num_embeddings=dataset.vocab_size,
            dim_embedding=args.dim_embedding,
            dim_hidden=args.dim_hidden, 
            num_layers=args.num_layers,
            bidirectional=args.bidirectional, 
            dim_latent=args.dim_latent, 
            word_dropout=args.word_dropout,
            dropout=args.dropout,
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            max_sequence_length=args.max_sequence_length).to(device)
    elif args.model == 'transformer':
        model = TransformerVAE(
            num_embeddings=dataset.vocab_size,
            dim_model=args.dim_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers,
            dim_latent=args.dim_latent, 
            word_dropout=args.word_dropout,
            dropout=args.dropout,
            sos_idx=dataset.sos_idx,
            eos_idx=dataset.eos_idx,
            pad_idx=dataset.pad_idx,
            unk_idx=dataset.unk_idx,
            max_sequence_length=args.max_sequence_length).to(device)
    else:
        raise ValueError

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))
    model.eval()

    ce_criterion = MaskedCrossEntropyLoss()
    kl_criterion = KLLoss()

    with torch.no_grad():
        for iteration, (input, target, length) in enumerate(loader):
            input = input.to(device)
            target = target.to(device)
            length = length.to(device)

            mean, logvar = model.encode(input, length)
            z = model.reparameterize(mean, logvar)
            output = model.infer(z)

            for i in range(len(target)):
                print('input : %s\noutput: %s\n' % (
                    decode_sentnece_from_token(input[i].tolist(), dataset.i2w),
                    decode_sentnece_from_token(output[i].tolist(), dataset.i2w)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=60)

    # model settings
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-dl', '--dim_latent', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.6)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)

    # rnn settings
    parser.add_argument('-de', '--dim_embedding', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-dh', '--dim_hidden', type=int, default=256)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    # transformer settings
    parser.add_argument('-dm', '--dim_model', type=int, default=256)
    parser.add_argument('-nh', '--nhead', type=int, default=4)
    parser.add_argument('-df', '--dim_feedforward', type=int, default=256)

    args = parser.parse_args()

    main(args)
