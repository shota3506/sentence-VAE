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

from dataset import QuoraDataset, collate_fn
from models.vae import VAE
from loss import MaskedCrossEntropyLoss, KLLoss

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


def main(args):
    question_file = os.path.join(args.data_dir, 'preprocessed_questions.json')
    vocab_file = os.path.join(args.data_dir, 'vocab.json')

    train_dataset = QuoraDataset(os.path.join(args.data_dir, 'train.json'), question_file, vocab_file)
    valid_dataset = QuoraDataset(os.path.join(args.data_dir, 'valid.json'), question_file, vocab_file)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory=torch.cuda.is_available())
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory=torch.cuda.is_available())

    sos_idx = train_dataset.vocab['w2i']['<sos>']
    eos_idx = train_dataset.vocab['w2i']['<eos>']
    pad_idx = train_dataset.vocab['w2i']['<pad>']
    unk_idx = train_dataset.vocab['w2i']['<unk>']
    model = VAE(
        rnn_type=args.rnn_type,
        num_embeddings=train_dataset.vocab_size,
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

    re_criterion = MaskedCrossEntropyLoss()
    kl_criterion = KLLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        tracker = {'ELBO': []}
        pbar = tqdm(train_loader)
        pbar.set_description('Epoch %3d / %d' % (epoch + 1, args.epochs))
        for iteration, (input, target, length) in enumerate(pbar):
            input = input.to(device)
            target = target.to(device)
            length = length.to(device)

            optimizer.zero_grad()
            output, mean, logvar, z = model(input, length, True)

            re_loss = re_criterion(output, target, target!=pad_idx)
            kl_loss = kl_criterion(mean, logvar)
            kl_weight = kl_anneal_function(args.anneal_function, step, args.k, args.x0)

            loss = re_loss + kl_weight * kl_loss
            elbo = re_loss + kl_loss

            loss.backward()
            optimizer.step()
            step+=1

            # bookkeepeing
            tracker['ELBO'].append(elbo.item())

            if iteration % args.print_every == 0 or iteration+1 == len(train_loader):
                pbar.set_postfix(loss=elbo.item(), re_loss=re_loss.item(), kl_loss=kl_loss.item(), kl_weight=kl_weight)

        print("Train Epoch %3d/%d, Mean ELBO %9.4f"%(epoch+1, args.epochs, sum(tracker['ELBO']) / len(tracker['ELBO'])))

        model.eval()
        tracker = {'ELBO': []}
        with torch.no_grad():
            for input, target, length in valid_loader:
                input = input.to(device)
                target = target.to(device)
                length = length.to(device)

                output, mean, logvar, z = model(input, length)

                re_loss = re_criterion(output, target, target!=pad_idx)
                kl_loss = kl_criterion(mean, logvar)

                elbo = re_loss + kl_loss

                # bookkeepeing
                tracker['ELBO'].append(elbo.item())

        print("Valid Epoch %3d/%d, Mean ELBO %9.4f"%(epoch+1, args.epochs, sum(tracker['ELBO']) / len(tracker['ELBO'])))

        checkpoint_path = os.path.join(args.save_path, "model-%d.pth" % (epoch+1))
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--max_sequence_length', type=int, default=50)

    # model settings
    parser.add_argument('-dl', '--dim_latent', type=int, default=64)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.75)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)

    # rnn settings
    parser.add_argument('-de', '--dim_embedding', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-dh', '--dim_hidden', type=int, default=256)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    # training settings
    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-save','--save_path', type=str, default='save')

    args = parser.parse_args()

    main(args)
