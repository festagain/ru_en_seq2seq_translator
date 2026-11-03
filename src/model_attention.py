import random

import torch
import torch.nn as nn

from .model_baseline import Encoder


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.encoder_projection = nn.Linear(
            hidden_dim, 
            attention_dim, 
            bias= False
        )
        self.decoder_projection = nn.Linear(
            hidden_dim,
            attention_dim,
            bias= False
        )
        self.energy_projection = nn.Linear(
            attention_dim,
            1,
            bias= False
        )

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        last_decoder_hidden = decoder_hidden[-1]
        energy = torch.tanh(
            self.encoder_projection(encoder_outputs) +
            self.decoder_projection(last_decoder_hidden).unsqueeze(1)
        )
        scores = self.energy_projection(energy).squeeze(-1)
        scores = scores.masked_fill(~source_mask, -1e9)
        attention_weights = torch.softmax(scores, dim = 1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        return context, attention_weights

class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers, pad_idx, attention_dim = None, dropout = 0.2):
        super().__init__()
        if attention_dim is None:
            attention_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden_dim, attention_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout= dropout if num_layers > 1 else 0
        )
        self.fc_out = nn.Linear(
            in_features= hidden_dim + hidden_dim + embed_dim,
            out_features= output_dim
        )
    def forward(self, input_token, hidden, cell, encoder_outputs, source_mask):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token)) # batch_size, hidden_dim
        context, attention_weights = self.attention(
            encoder_outputs,
            hidden,
            source_mask
        ) # batch_size, hidden_dim
        lstm_input = torch.cat(
            [embedded, context.unsqueeze(1)],
            dim=2
        )
        output, (hidden, cell) = self.lstm(
            lstm_input, 
            (hidden, cell)
        )
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction_input = torch.cat(
            [output, context, embedded],
            dim=1
        )
        prediction = self.fc_out(prediction_input)
        return prediction, hidden, cell, attention_weights

class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, source_lenghts, target, teacher_forcing_rate = 0.5):
        batch_size = source.size(0)
        target_length = target.size(1)
        target_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(
            batch_size,
            target_length,
            target_vocab_size,
            device=source.device
        )
        encoder_outputs, hidden, cell = self.encoder(source, source_lenghts)
        source_positions = torch.arange(
            source.size(1),
            device=source.device
        ).unsqueeze(0)

        source_mask = source_positions < source_lenghts.to(source.device).unsqueeze(1)

        input_token = target[:, 0]
        for token_idx in range(1, target_length):
            prediction, hidden, cell, _ = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs,
                source_mask
            )
            outputs[:, token_idx, :] = prediction
            predicted_token = prediction.argmax(dim = 1)
            use_teacher_forcing = (
                random.random() < teacher_forcing_rate
            )

            input_token = (
                target[:, token_idx] 
                if use_teacher_forcing 
                else predicted_token
            )
        return outputs