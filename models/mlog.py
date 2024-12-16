import json
import torch.nn as nn
import torch
from torch.nn import Parameter
from typing import *
import numpy as np
from enum import IntEnum
import torch.nn.functional as F



def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class Embedder(nn.Module):
    def __init__(
            self,
            embedding_dim,
            pretrain_matrix=None,
            freeze=False
    ):
        super(Embedder, self).__init__()
        if pretrain_matrix is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                pretrain_matrix, padding_idx=1, freeze=freeze
            )

    def forward(self, x):
        return torch.matmul(x, self.embedding_layer.weight)


class MLog(nn.Module):
    def __init__(self, input_sz, hidden_sz, cnn_length, in_size2=300, num_layers=2, mog_iteration=2, filter_num=3,
                 filter_sizes="3,4,5", max_length=100, pre_train=None):
        super(MLog, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.cnn_length = cnn_length
        self.mog_iterations = mog_iteration
        self.in_size2 = in_size2
        if pre_train:
            self.pre_train = torch.Tensor(np.array(read_json(pre_train)))

        self.embedder = Embedder(
            embedding_dim=self.input_size,
            pretrain_matrix=self.pre_train,
            freeze=True,
        )

        self.linear_in = nn.Linear(self.input_size, self.in_size2)
        self.lstm = nn.LSTM(input_sz,
                            hidden_sz,
                            num_layers=num_layers,
                            batch_first=True)
        self.Wih = Parameter(torch.Tensor(self.in_size2, hidden_sz * 4))
        self.Whh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bih = Parameter(torch.Tensor(hidden_sz * 4))
        self.bhh = Parameter(torch.Tensor(hidden_sz * 4))
        self.dropout = nn.Dropout(0.5)
        # Mogrifiers
        self.Q = Parameter(torch.Tensor(hidden_sz, self.in_size2))
        self.R = Parameter(torch.Tensor(self.in_size2, hidden_sz))
        self.fc = nn.Linear(self.hidden_size, 2)
        self.init_weights()

        self.filter_num = filter_num
        self.filter_sizes = [int(fsz) for fsz in filter_sizes.split(',')]
        self.convs = nn.ModuleList([nn.Conv2d(1, filter_num, (fsz, self.hidden_size)) for fsz in self.filter_sizes])
        self.linear = nn.Linear(len(self.filter_sizes) * filter_num, 2)
        self.identity = nn.Identity()  # 匿名层，用于LSTM输出，注册钩子函数
        self.identity2 = nn.Identity()  # 用于卷积层输出

    def init_weights(self):
        """
        Weight initialization: Xavier initialization for W, Q, and R; zero initialization for the bias b.
        :return:
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                # Ensure that the gradients are stable during both forward and backward propagation
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self, xt, ht):
        """
        calculate mogrify
        :param xt:
        :param ht:
        :return:
        """
        for i in range(1, self.mog_iterations + 1):
            if (i % 2 == 0):
                ht = (2 * torch.sigmoid(xt @ self.R) * ht)
            else:
                xt = (2 * torch.sigmoid(ht @ self.Q) * xt)
        return xt, ht

    def forward(self, features, device=None, init_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = features
        batch_sz, seq_sz, _ = x.size()
        self.seq_size = seq_sz
        x = torch.tanh(self.linear_in(x))
        hidden_seq = []
        if init_states is None:
            ht = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz, self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states

        for t in range(seq_sz):
            xt = x[:, t, :]
            # Feed the input vector and hidden state vector of the current time step into the mogrifier
            xt, ht = self.mogrify(xt, ht)
            gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            # LSTM
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            # outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)

            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        x = hidden_seq.cuda().view(-1, 1, hidden_seq.size(1), self.hidden_size)
        x = [F.relu(conv(x)) for conv in self.convs]
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = self.identity(torch.cat(x, 1))
        x = self.dropout(x)
        out = self.linear(x)

        return out
