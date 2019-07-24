import torch
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, rnn_units, seq_len, num_layers, dropout):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, rnn_units, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, rnn_units, seq_len, num_layers, dropout, output_size):
        super(Decoder, self).__init__()
        self.rnn_units = rnn_units

        self.attn = nn.Linear(self.rnn_units+1, seq_len)
        self.attn_combine = nn.Linear(self.rnn_units*seq_len+1, rnn_units)
        self.lstm_cell = nn.LSTMCell(rnn_units, rnn_units)
        self.linear = nn.Linear(rnn_units, output_size)

    def forward(self, x, h, c, enc_vec):
        batch_size = x.shape[0]
        attn_weights = F.softmax(
            torch.tanh(self.attn(torch.cat((x, h), 1))), dim=1).unsqueeze(2)
        attn = attn_weights * enc_vec
        x = torch.cat((attn.view(batch_size, -1), x), 1)
        x = F.relu(self.attn_combine(x))
        h, c = self.lstm_cell(x, (h, c))
        x = self.linear(h)
        return x, h, c

    def initHidden(self, batch_size):
        hx = torch.zeros(batch_size, self.rnn_units)
        cx = torch.zeros(batch_size, self.rnn_units)
        return hx, cx
