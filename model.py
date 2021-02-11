import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(
        self, 
        num_embeddings, 
        dim_embedding, 
        dim_hidden, 
        dim_latent, 
        num_layers, 
        bidirectional, 
        dropout,
        word_dropout, 
        dropped_index=3,
    ) -> None:
        super(VAE, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.dropout = nn.Dropout(p=dropout)
        self.word_dropout = WordDropout(p=word_dropout, dropped_index=dropped_index)
        
        self.embedding = nn.Embedding(num_embeddings, dim_embedding)
        self.encoder = nn.GRU(
            dim_embedding, dim_hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), bidirectional=self.bidirectional, batch_first=True)
        self.decoder = nn.GRU(
            dim_embedding, dim_hidden, num_layers=num_layers, dropout=(dropout if num_layers > 1 else 0), batch_first=True)

        self.fc_mean =  nn.Linear(dim_hidden * (2 if bidirectional else 1) * num_layers, dim_latent)
        self.fc_logvar =  nn.Linear(dim_hidden * (2 if bidirectional else 1) * num_layers, dim_latent)

        self.fc_hidden = nn.Linear(dim_latent, dim_hidden * num_layers)
        self.fc = nn.Linear(dim_hidden, num_embeddings)

    def forward(self, src, length):
        bsz = src.shape[0]
        tgt = src.clone()

        mean, logvar = self.encode(src, length)
        z = self.reparameterize(mean, logvar)

        hidden = self.fc_hidden(z)
        hidden = hidden.view(bsz, -1, self.dim_hidden).transpose(0, 1).contiguous()
        
        output, _ = self.decode(tgt, hidden)
        return output, mean, logvar, z

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def encode(self, src, length):
        bsz = src.shape[0]
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, length, enforce_sorted=False, batch_first=True)
        _, hidden = self.encoder(packed)

        # flatten hidden state
        hidden = hidden.transpose(0, 1).contiguous().view(bsz, -1)

        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

    def decode(self, tgt, hidden):
        tgt = self.word_dropout(tgt)
        embedded = self.dropout(self.embedding(tgt))
        output, state = self.decoder(embedded, hidden)
        output = self.fc(output)
        return output, state
    
    def step(self, last_predictions, state, timestep):
        last_predictions = last_predictions.unsqueeze(1)
        hidden = state['hidden'].permute(1, 0, 2).contiguous()

        output, hidden = self.decode(last_predictions, hidden)
        output = output.squeeze(1)
        log_probabilities = F.log_softmax(output, dim=-1)

        hidden = hidden.permute(1, 0, 2)
        return log_probabilities, {'hidden': hidden}

    # def infer(self, z):
        # device = z.device

        # hidden = self.fc_hidden(z)
        # hidden = hidden.view(1, -1, self.dim_hidden).transpose(0, 1).contiguous()

        # ys = torch.ones(1, 1).fill_(self.sos_idx).long().to(device)
        # input = torch.ones(1, 1).fill_(self.sos_idx).long().to(device)
        # for i in range(self.max_sequence_length):
            # embedded = self.embedding(input)
            # output, hidden = self.decoder_rnn(embedded, hidden)
            # output = self.fc(output)[-1]
            # _, next_word = torch.max(output, dim=1)
            # ys = torch.cat([ys, torch.ones(1, 1).fill_(next_word.item()).long().to(device)], dim=0)
            # input = torch.ones(1, 1).fill_(next_word.item()).long().to(device)

        # ys = ys[1:].transpose(0, 1)
            
        # return ys

class WordDropout(nn.Module):
    def __init__(
        self,
        p: float,
        dropped_index: int,
    ) -> None:
        super(WordDropout, self).__init__()
        self._p = p
        self._dropped_index = dropped_index
    
    def forward(self, x):
        if self.training and self._p > 0:
            rand = torch.rand_like(x, dtype=torch.float)
            mask = rand < self._p
            mask[:, 0] = False
            x[mask] = self._dropped_index
        return x