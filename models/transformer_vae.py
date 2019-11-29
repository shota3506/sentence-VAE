import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerVAE(nn.Module):
    def __init__(self, num_embeddings, dim_model, nhead, dim_feedforward, num_layers, dim_latent, word_dropout, dropout, 
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length):
        super(TransformerVAE, self).__init__()
        self.dim_model = dim_model

        self.mask = None

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.word_dropout_rate = word_dropout
        self.embedding = nn.Embedding(num_embeddings, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout)

        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(dim_model, nhead, dim_feedforward, dropout), num_layers)
        self.transformer_decoder = TransformerDecoder(
            TransformerDecoderLayer(dim_model, nhead, dim_feedforward, dropout), num_layers)

        self.fc_mean =  nn.Linear(dim_model, dim_latent)
        self.fc_logvar =  nn.Linear(dim_model, dim_latent)

        self.fc_hidden = nn.Linear(dim_latent, dim_model)
        self.fc = nn.Linear(dim_model, num_embeddings)
        
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _generate_key_padding_mask(self, length, max_sequence_length):
        mask = torch.arange(max_sequence_length, device=length.device).unsqueeze(0).repeat(len(length), 1)
        mask = mask >= length.unsqueeze(1)
        return mask

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def embed(self, src):
        embedded = self.embedding(src) * math.sqrt(self.dim_model)
        embedded = self.pos_encoder(embedded)
        return embedded
    
    def encode(self, src, length):
        device = src.device

        src = src.transpose(0, 1)
        key_padding_mask = self._generate_key_padding_mask(length, len(src)).to(device)
        embedded = self.embed(src)

        memory = self.transformer_encoder(embedded, src_key_padding_mask=key_padding_mask)
        memory = memory[0]

        mean = self.fc_mean(memory)
        logvar = self.fc_logvar(memory)
        return mean, logvar

    def decode(self, z, tgt, length, word_dropout=False):
        device = tgt.device

        memory = self.fc_hidden(z).unsqueeze(0)

        # randomly replace decoder input with <unk>
        if word_dropout and self.word_dropout_rate > 0:
            rand = torch.rand_like(tgt, dtype=torch.float)
            mask = (rand < self.word_dropout_rate) & (tgt != self.sos_idx) & (tgt != self.pad_idx)
            tgt[mask] = self.unk_idx

        tgt = tgt.transpose(0, 1)
        key_padding_mask = self._generate_key_padding_mask(length, len(tgt)).to(device)
        embedded = self.embed(tgt)
        if self.mask is None or self.mask.size(0) != len(tgt):
            self.mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
        output = self.transformer_decoder(embedded, memory, self.mask, tgt_key_padding_mask=key_padding_mask)
        output = self.fc(output)
        output = output.transpose(0, 1).contiguous()
        return output

    def forward(self, src, length, word_dropout=False):
        tgt = src.clone()
        mean, logvar = self.encode(src, length)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, tgt, length, word_dropout)

        return output, mean, logvar, z

    def infer(self, z):
        device = z.device
        memory = self.fc_hidden(z).unsqueeze(0)

        ys = torch.ones(1, 1).fill_(self.sos_idx).long().to(device)
        for i in range(self.max_sequence_length):
            tgt_embedded = self.embedding(ys) * math.sqrt(self.dim_model)
            tgt_embedded = self.pos_encoder(tgt_embedded)
            mask = self._generate_square_subsequent_mask(len(ys)).to(device)
            output = self.transformer_decoder(tgt_embedded, memory, mask)
            output = self.fc(output)[-1]
            _, next_word = torch.max(output, dim=1)
            ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word.item()).long().to(device)], dim=0)

        return ys.transpose(0, 1)
