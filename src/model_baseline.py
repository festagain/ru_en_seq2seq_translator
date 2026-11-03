import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers, pad_idx, dropout = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=input_dim, 
                                      embedding_dim=embed_dim, 
                                      padding_idx= pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size= embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers,
            batch_first= True,  
            dropout= dropout if num_layers > 1 else 0
        )
    def forward(self, source, source_lengths):
        embedded = self.dropout(self.embedding(source))
        packed = pack_padded_sequence(
            embedded,
            source_lengths.cpu(),
            batch_first= True,
            enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first= True
        )
        return outputs, hidden, cell
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, pad_idx, dropout= 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_embeddings=output_dim, 
                                      embedding_dim= embed_dim, 
                                      padding_idx= pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=embed_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers,
                            batch_first= True, 
                            dropout= dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1) # batch_size, 1
        embedded = self.dropout(self.embedding(input_token)) # batch_size, 1, embedd_dim
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell)) # batch_size, 1, hidden_dim
        prediction = self.fc_out(output.squeeze(1)) # batch_size, output_dim
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, sources_length, target, teacher_forcing_rate = 0.5):
        batch_size = source.size(0)
        target_lenght = target.size(1)
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(
            batch_size,
            target_lenght,
            target_vocab_size,
            device = source.device
        )
        _, hidden, cell = self.encoder(source, sources_length)
        input_token = target[:, 0]
        for token_idx in range(1, target_lenght):
            prediction, hidden, cell = self.decoder(
                input_token, 
                hidden,
                cell
            )
            outputs[:, token_idx, :] = prediction
            prediction_token = prediction.argmax(dim = 1)
            use_teacher_forcing = (
                random.random() < teacher_forcing_rate
            )
            input_token = (
                target[:, token_idx] if use_teacher_forcing else prediction_token
            )
        return outputs