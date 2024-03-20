import math
import argparse
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from search import BeamSearch
from dataset import SentenceDataset
from tokenizer import Tokenizer
from sentence_vae import VAE, LmCrossEntropyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_model_arguments(parser) -> None:
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--dim_hidden", type=int, default=512)
    parser.add_argument("--dim_latent", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")


def main():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers()

    # train command
    parser_train = subparser.add_parser("train")
    parser_train.set_defaults(func=train)

    add_model_arguments(parser_train)

    parser_train.add_argument("--train_file", type=str, required=True)
    parser_train.add_argument("--valid_file", type=str, required=True)
    parser_train.add_argument("--vocab_file", type=str, required=True)
    # train
    parser_train.add_argument("--dropout", type=float, default=0.1)
    parser_train.add_argument("--word_dropout", type=float, default=0.25)
    # optim
    parser_train.add_argument("--batch_size", type=int, default=128)
    parser_train.add_argument("--num_epochs", type=int, default=30)
    parser_train.add_argument("--learning_rate", type=float, default=0.001)
    parser_train.add_argument("--print_every", type=int, default=100)
    parser_train.add_argument(
        "--checkpoint_file", type=str, default="model.pth"
    )
    parser_train.add_argument("--log_file", type=str, default="train.log")
    parser_train.add_argument("--k", type=float, default=0.0025)
    parser_train.add_argument("--x0", type=int, default=2500)

    # sample command
    parser_sample = subparser.add_parser("sample")
    parser_sample.set_defaults(func=sample)

    add_model_arguments(parser_sample)

    parser_sample.add_argument("--vocab_file", type=str, required=True)
    parser_sample.add_argument(
        "--checkpoint_file", type=str, default="model.pth"
    )
    parser_sample.add_argument("--sample_size", type=int, default=10)
    parser_sample.add_argument("--search_width", type=int, default=1)

    args = parser.parse_args()
    args.func(args)


class KLAnnealer:
    def __init__(
        self,
        x0: int,
        k: float,
    ) -> None:
        self._step = 0
        self._x0 = x0
        self._k = k

    def __call__(self) -> float:
        return float(1 / (1 + math.exp(-self._k * (self._step - self._x0))))

    def step(self) -> None:
        self._step += 1


def train(args: argparse.Namespace):
    print(args.dim_embedding)

    logger = logging.getLogger(__name__)
    handler1 = logging.StreamHandler()
    handler1.setLevel(logging.INFO)
    handler2 = logging.FileHandler(filename=args.log_file, mode="w")
    handler2.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)8s %(message)s")
    )
    handler2.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    tokenizer = Tokenizer(args.vocab_file)
    train_dataset = SentenceDataset(args.train_file, tokenizer.encode)
    valid_dataset = SentenceDataset(args.valid_file, tokenizer.encode)
    train_loader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        drop_last=True,
    )

    model = VAE(
        num_embeddings=len(tokenizer),
        dim_embedding=args.dim_embedding,
        dim_hidden=args.dim_hidden,
        dim_latent=args.dim_latent,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=args.dropout,
        word_dropout=args.word_dropout,
        dropped_index=tokenizer.unk_index,
    ).to(device)

    annealer = KLAnnealer(x0=args.x0, k=args.k)

    criterion = LmCrossEntropyLoss(tokenizer.pad_index, reduction="batchmean")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09
    )

    logger.info("Start training")
    for epoch in range(args.num_epochs):
        (
            train_loss,
            train_ce_loss,
            train_kl_loss,
            valid_loss,
            valid_ce_loss,
            valid_kl_loss,
        ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pbar = tqdm(train_loader)
        pbar.set_description("[Epoch %d/%d]" % (epoch, args.num_epochs))

        # Train
        model.train()
        for itr, s in enumerate(pbar):
            beta = annealer()

            s = s.to(device)
            length = torch.sum(s != tokenizer.pad_index, dim=-1)
            output, mean, logvar, z = model(s, length)
            ce_loss = criterion(output[:, :-1, :], s[:, 1:])
            kl_loss = -0.5 * torch.mean(
                torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
            )
            loss = ce_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            annealer.step()

            train_loss += loss.item()
            train_ce_loss += ce_loss.item()
            train_kl_loss += kl_loss.item()
            if itr % args.print_every == 0:
                pbar.set_postfix(loss=train_loss / (itr + 1), beta=beta)
        train_loss /= len(train_loader)
        train_ce_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)

        # Valid
        model.eval()
        with torch.no_grad():
            for s in valid_loader:
                beta = annealer()

                s = s.to(device)
                length = torch.sum(s != tokenizer.pad_index, dim=-1)
                output, mean, logvar, z = model(s, length)
                ce_loss = criterion(output[:, :-1, :], s[:, 1:])
                kl_loss = -0.5 * torch.mean(
                    torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
                )
                loss = ce_loss + beta * kl_loss

                valid_loss += loss.item()
                valid_ce_loss += ce_loss.item()
                valid_kl_loss += kl_loss.item()
            valid_loss /= len(valid_loader)
            valid_ce_loss /= len(valid_loader)
            valid_kl_loss /= len(valid_loader)

        logger.info(
            "[Epoch %d/%d] Training loss: %.2f, CE loss: %.2f, KL loss: %.2f, "
            "Validation loss: %.2f, CE loss: %.2f, KL loss: %.2f"
            % (
                epoch,
                args.num_epochs,
                train_loss,
                train_ce_loss,
                train_kl_loss,
                valid_loss,
                valid_ce_loss,
                valid_kl_loss,
            )
        )

        torch.save(model.state_dict(), args.checkpoint_file)


def sample(args: argparse.Namespace):
    tokenizer = Tokenizer(args.vocab_file)

    searcher = BeamSearch(tokenizer.eos_index, beam_size=args.search_width)

    model = VAE(
        num_embeddings=len(tokenizer),
        dim_embedding=args.dim_embedding,
        dim_hidden=args.dim_hidden,
        dim_latent=args.dim_latent,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        dropout=0.0,
        word_dropout=0.0,
        dropped_index=tokenizer.unk_index,
    ).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint_file, map_location=device)
    )
    model.eval()

    z = torch.randn(args.sample_size, args.dim_latent, device=device)
    hidden = model.fc_hidden(z)
    hidden = (
        hidden.view(args.sample_size, -1, model.dim_hidden)
        .transpose(0, 1)
        .contiguous()
    )

    start_predictions = (
        torch.zeros(args.sample_size, device=device)
        .fill_(tokenizer.bos_index)
        .long()
    )
    start_state = {"hidden": hidden.permute(1, 0, 2)}
    predictions, log_probabilities = searcher.search(
        start_predictions, start_state, model.step
    )

    for pred in predictions:
        tokens = pred[0]
        tokens = tokens[tokens != tokenizer.eos_index].tolist()
        print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
