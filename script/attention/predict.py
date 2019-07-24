# Copyright © 2019 Aidemy inc. All Rights Reserved.
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import create_dataset
from model import Encoder, Decoder
import matplotlib.pyplot as plt

def calc_metric(pred, target):
    return np.sqrt(np.mean((target - pred)**2)), pearsonr(target.ravel(), pred.ravel())

def predict(region):
    np.random.seed(0)
    torch.manual_seed(0)

    input_len = 10
    encoder_units = 32
    decoder_units = 64
    encoder_rnn_layers = 3
    encoder_dropout = 0.2
    decoder_dropout = 0.2
    input_size = 2
    output_size = 1
    predict_len = 5
    batch_size = 16
    force_teacher = 0.8

    train_dataset, test_dataset, train_max, train_min = create_dataset(
        input_len, predict_len, region)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    enc = Encoder(input_size, encoder_units, input_len,
                  encoder_rnn_layers, encoder_dropout)
    dec = Decoder(encoder_units*2, decoder_units, input_len,
                  input_len, decoder_dropout, output_size)
    enc.load_state_dict(torch.load(f"models/{region}_enc.pth"))
    dec.load_state_dict(torch.load(f"models/{region}_dec.pth"))

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, drop_last=False)

    rmse = 0
    p = 4
    predicted = []
    true_target = []
    enc.eval()
    dec.eval()
    for encoder_input, decoder_input, target in test_loader:
        with torch.no_grad():
            enc_vec = enc(encoder_input)
            x = decoder_input[:, 0]
            h, c = dec.initHidden(1)
            pred = []
            for pi in range(predict_len):
                x, h, c = dec(x, h, c, enc_vec)
                pred += [x]
            pred = torch.cat(pred, dim=1)
            predicted += [pred[0, p].item()]
            true_target += [target[0, 0].item()]
    predicted = np.array(predicted).reshape(1, -1)
    predicted = (predicted + train_min) * (train_max - train_min)
    true_target = np.array(true_target).reshape(1, -1)
    true_target = (true_target + train_min) * (train_max - train_min)
    rmse, peasonr = calc_metric(predicted, true_target)
    print(f"{region} RMSE {rmse}")
    print(f"{region} r {peasonr[0]}")
    predicted = predicted.reshape(-1)
    true_target = true_target.reshape(-1)
    x = list(range(len(predicted)))
    plt.plot(x, predicted)
    plt.plot(x, true_target)
    plt.show()
    return f"{region} RMSE {rmse} r {peasonr[0]}"


if __name__ == "__main__":
    regions = ["New York", "oregon", "Illinois",
               "California", "Texas", "georgia"]
    predict(regions[0])
