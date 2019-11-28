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

# from ptb import PTB
from dataset import ParaphraseDataset, collate_fn
from models.vae import VAE
from loss import MaskedCrossEntropyLoss, KLLoss

device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def kl_anneal_function(anneal_function, step, k, x0):
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


def main(args):
    train_dataset = ParaphraseDataset(
        data_dir=args.data_dir,
        split='train',
        max_sequence_length=args.max_sequence_length)
    val_dataset = ParaphraseDataset(
        data_dir=args.data_dir,
        split='valid',
        max_sequence_length=args.max_sequence_length)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cpu_count(),
        pin_memory=torch.cuda.is_available())

    model = VAE(
        vocab_size=train_dataset.vocab_size,
        sos_idx=train_dataset.sos_idx,
        eos_idx=train_dataset.eos_idx,
        pad_idx=train_dataset.pad_idx,
        unk_idx=train_dataset.unk_idx,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        dropout=args.dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional).to(device)

    
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
        tracker = {'ELBO': [], 'z': []}
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
    parser.add_argument('--min_occ', type=int, default=1)

    parser.add_argument('-ep', '--epochs', type=int, default=30)
    parser.add_argument('-bs', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=64)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0.6)
    parser.add_argument('-do', '--dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-save','--save_model_path', type=str, default='save')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
