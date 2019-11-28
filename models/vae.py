import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False):
        super(VAE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.dropout_layer = nn.Dropout(p=dropout)
        
        self.encoder_rnn = nn.GRU(
            embedding_size, hidden_size, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = nn.GRU(
            embedding_size, hidden_size, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), batch_first=True)

        self.fc_mean =  nn.Linear(hidden_size * (2 if bidirectional else 1) * num_layers, latent_size)
        self.fc_logvar =  nn.Linear(hidden_size * (2 if bidirectional else 1) * num_layers, latent_size)

        self.fc_hidden = nn.Linear(latent_size, hidden_size * num_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std

    def encode(self, x, length):
        batch_size = x.shape[0]

        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        _, hidden = self.encoder_rnn(packed)

        # flatten hidden state
        hidden = hidden.transpose(0, 1).contiguous().view(batch_size, -1)


        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

    def decode(self, z, x, length, word_dropout=False):
        batch_size = len(z)

        hidden = self.fc_hidden(z)
        hidden = hidden.view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()

        tgt = x.clone()
        # randomly replace decoder input with <unk>
        if word_dropout and self.word_dropout_rate > 0:
            rand = torch.rand_like(x, dtype=torch.float)
            mask = (rand < self.word_dropout_rate) & (x != self.sos_idx) & (x != self.pad_idx)
            tgt[mask] = self.unk_idx

        embedded = self.embedding(tgt)
        embedded = self.dropout_layer(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        output, _ = self.decoder_rnn(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.fc(output)
        return output
    
    def forward(self, x, length, word_dropout=False):
        mean, logvar = self.encode(x, length)
        z = self.reparameterize(mean, logvar)
        output = self.decode(z, x, length, word_dropout)

        return output, mean, logvar, z

    def infer(self, z):
        device = z.device
        batch_size = len(z)

        hidden = self.fc_hidden(z)
        hidden = hidden.view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
        
        mask = torch.ones(batch_size, device=device).bool()
        sequence = torch.zeros(batch_size, self.max_sequence_length, device=device, dtype=torch.long).fill_(self.pad_idx)

        t = 0
        input = torch.zeros(batch_size, device=device, dtype=torch.long).fill_(self.sos_idx)
        while(t < self.max_sequence_length):
            embedded = self.embedding(input.unsqueeze(1))
            output, hidden = self.decoder_rnn(embedded, hidden)
            output = self.fc(output)
            sample = self._sample(output)

            sequence[:, t][mask] = sample[mask]
            mask = mask & (sample != self.eos_idx)
            input = sample

            t += 1
        return sequence

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample
