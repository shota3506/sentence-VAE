import torch
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from dataset import ParaphraseDataset, collate_fn


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


def decode_sentnece_from_token(tokens, i2w):
    sentence = ""
    for t in tokens:
        w = i2w[t]
        if w == "<pad>" or w == "<eos>":
            break
        sentence += " " + w
    return sentence


def expierment_name(args, ts):
    exp_name = str()
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts

    return exp_name
