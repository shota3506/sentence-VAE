import os
import json
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict
from tqdm import tqdm

from dataset import ParaphraseDataset, collate_fn
from models.vae import VAE
from loss import MaskedCrossEntropyLoss, KLLoss
from utils import decode_sentnece_from_token

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def main(args):

    dataset = ParaphraseDataset(
        data_dir=args.data_dir,
        split='test',
        max_sequence_length=args.max_sequence_length)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available())

    model = VAE(
        vocab_size=dataset.vocab_size,
        sos_idx=dataset.sos_idx,
        eos_idx=dataset.eos_idx,
        pad_idx=dataset.pad_idx,
        unk_idx=dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional).to(device)

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
                print('target: %s\noutput: %s\n' % (
                    decode_sentnece_from_token(target[i].tolist(), dataset.i2w, dataset.eos_idx),
                    decode_sentnece_from_token(output[i].tolist(), dataset.i2w, dataset.eos_idx)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=60)

    parser.add_argument('-bs', '--batch_size', type=int, default=32)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.5)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)

    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-save','--save_model_path', type=str, default='save')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
