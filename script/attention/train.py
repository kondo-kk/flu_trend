import pandas as pd
import numpy as np
from fastprogress import master_bar, progress_bar
from scipy.stats import pearsonr
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from adabound import AdaBound

from data import create_dataset
from model import Encoder, Decoder

in_dir = "./"
out_dir = "./"


def quantile_loss(pred, target, gamma=0.8):
    return torch.mean(
        torch.where(pred > target,
                    (1 - gamma) * (target-pred)**2,
                    gamma * (target - pred)**2))


def calc_metric(pred, target):
    return np.sqrt(np.mean((target - pred)**2)), pearsonr(target.ravel(), pred.ravel())


def train(region):
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
    epochs = 500
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

    optimizer = AdaBound(list(enc.parameters()) +
                         list(dec.parameters()), 0.01, final_lr=0.1)
    # optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), 0.01)
    criterion = nn.MSELoss()

    mb = master_bar(range(epochs))
    for ep in mb:
        train_loss = 0
        enc.train()
        dec.train()
        for encoder_input, decoder_input, target in progress_bar(train_loader, parent=mb):
            optimizer.zero_grad()
            enc_vec = enc(encoder_input)
            h = enc_vec[:, -1, :]
            _, c = dec.initHidden(batch_size)
            x = decoder_input[:, 0]
            pred = []
            for pi in range(predict_len):
                x, h, c = dec(x, h, c, enc_vec)
                rand = np.random.random()
                pred += [x]
                if rand < force_teacher:
                    x = decoder_input[:, pi]
            pred = torch.cat(pred, dim=1)
            # loss = quantile_loss(pred, target)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        test_loss = 0
        enc.eval()
        dec.eval()
        for encoder_input, decoder_input, target in progress_bar(test_loader, parent=mb):
            with torch.no_grad():
                enc_vec = enc(encoder_input)
                h = enc_vec[:, -1, :]
                _, c = dec.initHidden(batch_size)
                x = decoder_input[:, 0]
                pred = []
                for pi in range(predict_len):
                    x, h, c = dec(x, h, c, enc_vec)
                    pred += [x]
                pred = torch.cat(pred, dim=1)
            # loss = quantile_loss(pred, target)
            loss = criterion(pred, target)
            test_loss += loss.item()
        print(
            f"Epoch {ep} Train Loss {train_loss/len(train_loader)} Test Loss {test_loss/len(test_loader)}")

    if not os.path.exists("models"):
        os.mkdir("models")
    torch.save(enc.state_dict(), f"models/{region}_enc.pth")
    torch.save(dec.state_dict(), f"models/{region}_dec.pth")

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, drop_last=False)

    rmse = 0
    p = 0
    predicted = []
    true_target = []
    enc.eval()
    dec.eval()
    for encoder_input, decoder_input, target in progress_bar(test_loader, parent=mb):
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
            true_target += [target[0, p].item()]
    predicted = np.array(predicted).reshape(1, -1)
    predicted = predicted * (train_max - train_min) + train_min
    true_target = np.array(true_target).reshape(1, -1)
    true_target = true_target * (train_max - train_min) + train_min
    rmse, peasonr = calc_metric(predicted, true_target)
    print(f"{region} RMSE {rmse}")
    print(f"{region} r {peasonr[0]}")
    return f"{region} RMSE {rmse} r {peasonr[0]}"


if __name__ == '__main__':
    regions = ["New York", "oregon", "Illinois",
               "California", "Texas", "georgia"]
    results = []
    for i in range(6):

        results += [train(regions[i])]
    print(results)
    print("\n".join(results))
