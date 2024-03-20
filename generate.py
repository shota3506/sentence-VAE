import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from search import BeamSearch
from dataset import SentenceDataset
from tokenizer import Tokenizer
from model import VAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
# Data
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--output_file", type=str, default="output.txt")
parser.add_argument("--vocab_file", type=str, required=True)
# Model
parser.add_argument("--dim_embedding", type=int, default=256)
parser.add_argument("--dim_hidden", type=int, default=512)
parser.add_argument("--dim_latent", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--checkpoint_file", type=str, default="model.pth")
# Search
parser.add_argument("--search_width", type=int, default=1)

args = parser.parse_args()


def main() -> None:
    tokenizer = Tokenizer(args.vocab_file)
    dataset = SentenceDataset(args.input_file, tokenizer=tokenizer.encode)
    loader = DataLoader(
        dataset,
        args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

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
    model.load_state_dict(torch.load(args.checkpoint_file, map_location=device))
    model.eval()

    print("Generating sentence...")
    all_hypotheses = []
    with torch.no_grad():
        for s in tqdm(loader):
            s = s.to(device)
            length = torch.sum(s != tokenizer.pad_index, dim=-1)
            bsz = s.shape[0]

            mean, logvar = model.encode(s, length)
            # z = model.reparameterize(mean, logvar)
            z = mean

            hidden = model.fc_hidden(z)
            hidden = hidden.view(bsz, -1, model.dim_hidden).transpose(0, 1).contiguous()

            start_predictions = (
                torch.zeros(bsz, device=device).fill_(tokenizer.bos_index).long()
            )
            start_state = {"hidden": hidden.permute(1, 0, 2)}
            predictions, log_probabilities = searcher.search(
                start_predictions, start_state, model.step
            )

            for preds in predictions:
                tokens = preds[0]
                tokens = tokens[tokens != tokenizer.eos_index].tolist()
                all_hypotheses.append(tokenizer.decode(tokens))
    print("Done")

    with open(args.output_file, "w") as f:
        f.write("\n".join(all_hypotheses))


if __name__ == "__main__":
    main()
