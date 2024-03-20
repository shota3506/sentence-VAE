import argparse

import torch

from search import BeamSearch
from tokenizer import Tokenizer
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--vocab_file", type=str, required=True)
# Model
parser.add_argument("--dim_embedding", type=int, default=256)
parser.add_argument("--dim_hidden", type=int, default=512)
parser.add_argument("--dim_latent", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--checkpoint_file", type=str, default="model.pth")
# Search
parser.add_argument("--search_width", type=int, default=1)

args = parser.parse_args()


def main() -> None:
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

    sentence1 = input("Please input sentence1: ")
    sentence2 = input("Please input sentence2: ")

    s1 = (
        [tokenizer.bos_index]
        + tokenizer.encode(sentence1)
        + [tokenizer.eos_index]
    )
    s2 = (
        [tokenizer.bos_index]
        + tokenizer.encode(sentence2)
        + [tokenizer.eos_index]
    )

    z1, _ = model.encode(
        torch.tensor([s1]).to(device), torch.tensor([len(s1)]).to(device)
    )
    z2, _ = model.encode(
        torch.tensor([s2]).to(device), torch.tensor([len(s2)]).to(device)
    )

    print("\nGenerate intermediate sentences")
    print("      %s" % sentence1)
    for r in range(1, 10):
        z = (1 - 0.1 * r) * z1 + 0.1 * r * z2
        hidden = model.fc_hidden(z)
        hidden = (
            hidden.view(1, -1, model.dim_hidden).transpose(0, 1).contiguous()
        )

        start_predictions = (
            torch.zeros(1, device=device).fill_(tokenizer.bos_index).long()
        )
        start_state = {"hidden": hidden.permute(1, 0, 2)}
        predictions, log_probabilities = searcher.search(
            start_predictions, start_state, model.step
        )

        tokens = predictions[0, 0]
        tokens = tokens[tokens != tokenizer.eos_index].tolist()
        print("[%d:%d] %s" % (10 - r, r, tokenizer.decode(tokens)))
    print("      %s" % sentence2)


if __name__ == "__main__":
    main()
