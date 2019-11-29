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
from models.rnn_vae import RNNVAE
from models.transformer_vae import TransformerVAE
from loss import MaskedCrossEntropyLoss, KLLoss

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


def load_dataset(data_dir, split, batch_size):
    dataset = ParaphraseDataset(data_dir=data_dir, split=split)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split=='train'),
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available())
    return dataset, loader


def main(args):
    train_dataset, train_loader = load_dataset(args.data_dir, 'train', args.batch_size)
    val_dataset, val_loader = load_dataset(args.data_dir, 'valid', args.batch_size)

    if args.model == 'rnn':
        model = RNNVAE(
            rnn_type=args.rnn_type,
            num_embeddings=train_dataset.vocab_size,
            dim_embedding=args.dim_embedding,
            dim_hidden=args.dim_hidden, 
            num_layers=args.num_layers,
            bidirectional=args.bidirectional, 
            dim_latent=args.dim_latent, 
            word_dropout=args.word_dropout,
            dropout=args.dropout,
            sos_idx=train_dataset.sos_idx,
            eos_idx=train_dataset.eos_idx,
            pad_idx=train_dataset.pad_idx,
            unk_idx=train_dataset.unk_idx,
            max_sequence_length=args.max_sequence_length).to(device)
    elif args.model == 'transformer':
        model = TransformerVAE(
            num_embeddings=train_dataset.vocab_size,
            dim_model=args.dim_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers,
            dim_latent=args.dim_latent, 
            word_dropout=args.word_dropout,
            dropout=args.dropout,
            sos_idx=train_dataset.sos_idx,
            eos_idx=train_dataset.eos_idx,
            pad_idx=train_dataset.pad_idx,
            unk_idx=train_dataset.unk_idx,
            max_sequence_length=args.max_sequence_length).to(device)
    else:
        raise ValueError

    print(model)

    save_model_path = os.path.join(args.save_model_path)

    ce_criterion = MaskedCrossEntropyLoss()
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

            re_loss = ce_criterion(output, target, target!=train_dataset.pad_idx)
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

        checkpoint_path = os.path.join(save_model_path, "model-%d.pth" % (epoch+1))
        torch.save(model.state_dict(), checkpoint_path)

        model.eval()
        tracker = {'ELBO': []}
        with torch.no_grad():
            for iteration, (input, target, length) in enumerate(val_loader):
                input = input.to(device)
                target = target.to(device)
                length = length.to(device)

                output, mean, logvar, z = model(input, length)

                re_loss = ce_criterion(output, target, target!=train_dataset.pad_idx)
                kl_loss = kl_criterion(mean, logvar)

                elbo = re_loss + kl_loss

                # bookkeepeing
                tracker['ELBO'].append(elbo.item())

        print("Valid Epoch %3d/%d, Mean ELBO %9.4f"%(epoch+1, args.epochs, sum(tracker['ELBO']) / len(tracker['ELBO'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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

    # training settings
    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)
    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-save','--save_model_path', type=str, default='save')

    args = parser.parse_args()

    main(args)
