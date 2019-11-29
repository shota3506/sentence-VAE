import torch
import torch.nn as nn


class RNNVAE(nn.Module):
    def __init__(self, rnn_type, num_embeddings, dim_embedding, dim_hidden, num_layers, bidirectional, dim_latent, word_dropout, dropout,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length):
        super(RNNVAE, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.word_dropout_rate = word_dropout
        self.embedding = nn.Embedding(num_embeddings, dim_embedding)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.encoder_rnn = nn.GRU(
            dim_embedding, dim_hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = nn.GRU(
            dim_embedding, dim_hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), batch_first=True)

        self.fc_mean =  nn.Linear(dim_hidden * (2 if bidirectional else 1) * num_layers, dim_latent)
        self.fc_logvar =  nn.Linear(dim_hidden * (2 if bidirectional else 1) * num_layers, dim_latent)

        self.fc_hidden = nn.Linear(dim_latent, dim_hidden * num_layers)
        self.fc = nn.Linear(dim_hidden, num_embeddings)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def encode(self, src, length):
        batch_size = src.shape[0]

        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        _, hidden = self.encoder_rnn(packed)

        # flatten hidden state
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)

        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

    def decode(self, z, tgt, length, word_dropout=False):
        batch_size = len(z)

        hidden = self.fc_hidden(z)
        hidden = hidden.view(batch_size, -1, self.dim_hidden).transpose(0, 1).contiguous()

        # randomly replace decoder input with <unk>
        if word_dropout and self.word_dropout_rate > 0:
            rand = torch.rand_like(tgt, dtype=torch.float)
            mask = (rand < self.word_dropout_rate) & (tgt != self.sos_idx) & (tgt != self.pad_idx)
            tgt[mask] = self.unk_idx

        embedded = self.embedding(tgt)
        embedded = self.dropout_layer(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        output, _ = self.decoder_rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.fc(output)
        return output
    
    def forward(self, src, length, word_dropout=False):
        tgt = src.clone()
        mean, logvar = self.encode(src, length)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, tgt, length, word_dropout)

        return output, mean, logvar, z

    def infer(self, z):
        device = z.device

        hidden = self.fc_hidden(z)
        hidden = hidden.view(1, -1, self.dim_hidden).transpose(0, 1).contiguous()

        ys = torch.ones(1, 1).fill_(self.sos_idx).long().to(device)
        input = torch.ones(1, 1).fill_(self.sos_idx).long().to(device)
        for i in range(self.max_sequence_length):
            embedded = self.embedding(input)
            output, hidden = self.decoder_rnn(embedded, hidden)
            output = self.fc(output)[-1]
            _, next_word = torch.max(output, dim=1)
            ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word.item()).long().to(device)], dim=0)
            input = torch.ones(1, 1).fill_(next_word.item()).long().to(device)
            
        return ys.transpose(0, 1)
