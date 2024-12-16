#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import models.base_model as base_model
import json
import math


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


class logrobust(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_directions, num_keys, window_size, vocab_size,
                 freeze=True, pre_train=None):
        super(logrobust, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        if pre_train:
            self.pre_train = torch.Tensor(np.array(read_json(pre_train)))

        self.embedder = base_model.Embedder(
            vocab_size=self.vocab_size,
            embedding_dim=self.input_size,
            pretrain_matrix=self.pre_train,
            freeze=freeze,
            use_tfidf=True,
        )

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            bidirectional=(self.num_directions == 2),
        )

        self.attn = Attention(self.hidden_size * self.num_directions, self.window_size)
        self.fc = nn.Linear(self.hidden_size * self.num_directions, num_keys)

    def forward(self, features, device=None):
        input = features
        outputs, _ = self.rnn(input.float())
        representation = self.attn(outputs)
        logits = self.fc(representation)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        return logits


class Attention(nn.Module):
    def __init__(self, input_size, max_seq_len):
        super(Attention, self).__init__()
        self.atten_w = nn.Parameter(torch.randn(max_seq_len, input_size, 1))
        self.atten_bias = nn.Parameter(torch.randn(max_seq_len, 1, 1))
        self.glorot(self.atten_w)
        self.zeros(self.atten_bias)

    def forward(self, lstm_input):
        input_tensor = lstm_input.transpose(1, 0)
        input_tensor = (
                torch.bmm(input_tensor, self.atten_w) + self.atten_bias)
        input_tensor = input_tensor.transpose(1, 0)
        atten_weight = input_tensor.tanh()

        weighted_sum = torch.bmm(atten_weight.transpose(1, 2), lstm_input).squeeze()

        return weighted_sum

    def glorot(self, tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def zeros(self, tensor):
        if tensor is not None:
            tensor.data.fill_(0)
