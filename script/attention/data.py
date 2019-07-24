# Copyright © 2019 Aidemy inc. All Rights Reserved.
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

in_dir = "input/"
out_dir = "models/"


def create_encoder_input(dataset, pred_len=1, encoder_input_len=15, rag=1, num_order=0):
    encoder_input_data = []
    for i in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        data = [dataset[i + j, :] for j in range(encoder_input_len)]
        encoder_input_data += [data]
    encoder_input_data = np.array(encoder_input_data)
    return encoder_input_data


def create_decoder_input(dataset, pred_len=1, encoder_input_len=15, rag=1):
    decoder_input_data = []
    for i in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        add_array = [dataset[i+j+rag+encoder_input_len, 1]
                     for j in range(pred_len)]
        decoder_input_data.append(add_array)
    decoder_input_data = np.array(decoder_input_data)
    decoder_input_data = decoder_input_data.reshape(
        decoder_input_data.shape[0], decoder_input_data.shape[1], 1)
    return decoder_input_data

def create_target_data(dataset, pred_len=1, encoder_input_len=15, rag=1):
    target_data = []
    for i in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        add_array = [[dataset[i+j+rag+encoder_input_len, 1]]
                     for j in range(pred_len)]
        target_data.append(add_array)
    target_data = np.array(target_data)
    return target_data


def transform(x):
    x = x + (np.random.rand()*0.2 - 0.1)
    return x


class FluDataset(Dataset):
    def __init__(self, encoder_input, decoder_input, target_data, transform=None):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input.astype("float32")
        self.target_data = target_data
        self.transform = transform

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, index):
        ei = self.encoder_input[index]
        di = self.decoder_input[index]
        t = self.target_data[index, :, 0]
        if self.transform is not None:
            ei = self.transform(ei)
            di = self.transform(di)
        return ei, di, t


def create_dataset(input_len, fit_len, region):
    df = pd.read_csv(in_dir+"df_{}_2010-2018.csv".format(region), index_col=0)
    dataset = df.values
    dataset = dataset.astype("float32")
    train_size = int(len(dataset)*0.67)
    train_ds = dataset[0:train_size, :]
    test_ds = dataset[train_size:len(dataset), :]
    train_max = np.amax(train_ds, axis=0)[1]
    train_min = np.amin(train_ds, axis=0)[1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_train = scaler.fit(train_ds)
    train_ds = scaler_train.transform(train_ds)
    test_ds = scaler_train.transform(test_ds)

    train_encoder_input_data = create_encoder_input(
        train_ds, encoder_input_len=input_len, pred_len=fit_len)
    train_target_data = create_target_data(
        train_ds, encoder_input_len=input_len, pred_len=fit_len)
    train_decoder_input_data = create_decoder_input(
        train_ds, encoder_input_len=input_len, pred_len=fit_len)

    test_encoder_input_data = create_encoder_input(
        test_ds, encoder_input_len=input_len, pred_len=fit_len)
    test_target_data = create_target_data(
        test_ds, encoder_input_len=input_len, pred_len=fit_len)
    test_decoder_input_data = create_decoder_input(
        test_ds, encoder_input_len=input_len, pred_len=fit_len)

    train_dataset = FluDataset(
        train_encoder_input_data, train_decoder_input_data, train_target_data, transform)
    test_dataset = FluDataset(test_encoder_input_data,
                              test_decoder_input_data, test_target_data)
    return train_dataset, test_dataset, train_max, train_min
