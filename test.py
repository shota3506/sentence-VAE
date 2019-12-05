import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

from dataset import QuoraDataset, collate_fn
from models.vae import VAE
from loss import MaskedCrossEntropyLoss, KLLoss
from utils import decode_sentnece_from_token

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def main(args):
    question_file = os.path.join(args.data_dir, 'preprocessed_questions.json')
    vocab_file = os.path.join(args.data_dir, 'vocab.json')

    dataset = QuoraDataset(os.path.join(args.data_dir, 'test.json'), question_file, vocab_file)
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory=torch.cuda.is_available())

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

    ce_criterion = MaskedCrossEntropyLoss()
    kl_criterion = KLLoss()

    model.eval()    
    with torch.no_grad():
        for input, target, length in loader:
            input = input.to(device)
            target = target.to(device)
            length = length.to(device)

            mean, logvar = model.encode(input, length)
            z = model.reparameterize(mean, logvar)
            ys = model.infer(z)

            print("-" * 100 + "\n")
            print("Source: %s\nOutput: %s\n" \
                 % (decode_sentnece_from_token(target[0].tolist(), dataset.vocab['i2w'], eos_idx),
                 decode_sentnece_from_token(ys[0].tolist(), dataset.vocab['i2w'], eos_idx)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)

    parser.add_argument('--data_dir', type=str, default='data')
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
