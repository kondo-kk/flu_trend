# Copyright © 2019 Aidemy inc. All Rights Reserved.
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, RNN, Dense, LSTMCell, LSTM
import math
from sklearn.metrics import mean_squared_error
in_dir = "./"
out_dir = "./"


# create_encoder_input_data
def create_encoder_input(dataset, pred_len=1, encoder_input_len=15, rag=1, num_order=0):
    encoder_input_data = []
    for j in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        add_array = [np.array([dataset[i+j+rag, s] for s in range(0, num_order*2+1, 2)]+[dataset[i+j, t]
                                                                                         for t in range(1, num_order*2+3, 2)]) for i in range(encoder_input_len)]
        encoder_input_data.append(add_array)
    encoder_input_data = np.array(encoder_input_data)
    return encoder_input_data

# create_decoder_input


def create_decoder_input(dataset, pred_len=1, encoder_input_len=15, rag=1):
    decoder_input_data = []
    for j in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        add_array = [0]
        if pred_len > 1:
            add_array.extend([dataset[i+j+rag+encoder_input_len, 1]
                              for i in range(pred_len-1)])
        decoder_input_data.append(add_array)
    decoder_input_data = np.array(decoder_input_data)
    decoder_input_data = decoder_input_data.reshape(
        decoder_input_data.shape[0], decoder_input_data.shape[1], 1)
    return decoder_input_data

# create_target_data


def create_target_data(dataset, pred_len=1, encoder_input_len=15, rag=1):
    target_data = []
    for j in range(len(dataset)-encoder_input_len-pred_len-rag+1):
        add_array = [[dataset[i+j+rag+encoder_input_len, 1]]
                     for i in range(pred_len)]
        target_data.append(add_array)
    target_data = np.array(target_data)
    return target_data


# definite loss function
def huber_loss(y_true, y_pred, clip_delta=1.0):

    error = y_true-y_pred
    cond = tf.keras.backend.abs(error) < clip_delta
    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error)-0.5*clip_delta)

    return tf.where(cond, squared_loss, linear_loss)


def huber_loss_mean(y_true, y_pred, clip_delta=1.0):
    return tf.keras.backend.mean(huber_loss(y_true, y_pred, clip_delta))


def main(region):
    scores = []
    df = pd.read_csv(in_dir+"df_{}_2010-2018.csv".format(region), index_col=0)
    dataset = df.values

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

    input_len = 15
    pred_len = 1

    train_encoder_input_data = create_encoder_input(
        train_ds, encoder_input_len=input_len, pred_len=pred_len)
    train_target_data = create_target_data(
        train_ds, encoder_input_len=input_len, pred_len=pred_len)
    train_decoder_input_data = create_decoder_input(
        train_ds, encoder_input_len=input_len, pred_len=pred_len)

    test_encoder_input_data = create_encoder_input(
        test_ds, encoder_input_len=input_len, pred_len=pred_len)
    test_target_data = create_target_data(
        test_ds, encoder_input_len=input_len, pred_len=pred_len)
    test_decoder_input_data = create_decoder_input(
        test_ds, encoder_input_len=input_len, pred_len=pred_len)

    layer = 256

    r_dropout_en = 0.1
    r_dropout_de = 0.1
    batch_size = 32
    parameters = [layer, r_dropout_en, r_dropout_de, batch_size, input_len]
    num_output_features = 1
    num_input_features = 2

    encoder_inputs = Input(shape=(None, num_input_features), name="en_input")
    encoder_cells = []

    encoder_cells.append(LSTMCell(layer, recurrent_dropout=r_dropout_en))
    encoder = RNN(encoder_cells, return_state=True)
    encoder_outputs_states = encoder(encoder_inputs)
    encoder_states = encoder_outputs_states[1:]

    decoder_inputs = Input(shape=(None, 1), name="de_input")
    decoder = LSTM(layer, return_sequences=True,
                   return_state=True, recurrent_dropout=r_dropout_de)
    decoder_outputs_states = decoder(
        decoder_inputs, initial_state=encoder_states)
    decoder_outputs = decoder_outputs_states[0]
    decoder_dense = Dense(1, activation="relu")
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer="adam", loss=huber_loss_mean)
    # model.compile(optimizer="adam",loss="mse")

    import matplotlib.pyplot as plt
    history = model.fit([train_encoder_input_data, train_decoder_input_data], train_target_data,
                        batch_size=batch_size,
                        epochs=30,
                        verbose=0,
                        validation_data=([test_encoder_input_data, test_decoder_input_data], test_target_data))
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'])
    plt.plot(history.history["val_loss"])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', "test"], loc='upper left')
    plt.show()

    test_predict_data = model.predict(
        [test_encoder_input_data, test_decoder_input_data])
    test_predict_data_cvt = test_predict_data.reshape(
        test_predict_data.shape[0])
    test_predict_plot = test_predict_data_cvt*(train_max-train_min)+train_min

    test_actual_plot = dataset[len(train_ds)+input_len+pred_len:, 1]

    test_score = math.sqrt(mean_squared_error(
        test_actual_plot, test_predict_plot))
    scores.append(test_score)
    print('Test Score: %.2f RMSE' % (test_score))

    df_output = pd.DataFrame({"predict": test_predict_plot,
                              "actual": test_actual_plot})

    df_output.to_csv(out_dir+"df_seq2seq_{}_1week.csv".format(region))

    fig1 = plt.figure(figsize=(12, 4))
    plt.title("{}:test_plot".format(region))
    ax = fig1.add_subplot(1, 1, 1)
    ax.set_xlabel("Time(week)")
    ax.set_ylabel("Unweighted ILI(%)")
    ax.plot(test_predict_plot, label="test_pred")
    ax.plot(test_actual_plot, label="test_act")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()

    encoder_predict_model = Model(encoder_inputs, encoder_states)
    layers = [layer]
    decoder_states_inputs = []
    for neurons in layers[::-1]:
        decoder_states_inputs.append(Input(shape=(neurons,), name="de_in_h"))
        decoder_states_inputs.append(Input(shape=(neurons,), name="de_in_c"))
    decoder_outputs_states = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = decoder_outputs_states[1:]
    decoder_outputs = decoder_outputs_states[0]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_predict_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    for pred_len in range(2, 6):
        test_encoder_input_data = create_encoder_input(
            test_ds, encoder_input_len=input_len, pred_len=pred_len)
        y_predicted = []
        states = encoder_predict_model.predict(test_encoder_input_data)
        decoder_input = np.zeros((test_encoder_input_data.shape[0], 1, 1))
        for _ in range(pred_len):
            outputs_states = decoder_predict_model.predict(
                [decoder_input]+states)
            output = outputs_states[0]
            states = outputs_states[1:]
            decoder_input = output
            y_predicted.append(output)
        y_predicted = np.concatenate(y_predicted, axis=1)[
            :-(pred_len-1)][:, :, 0]
        y_predicted_cvt = y_predicted*(train_max-train_min)+train_min
        y_predicted_plot = y_predicted_cvt[:, pred_len-1]
        test_target_data_plot = dataset[len(train_ds)+input_len+pred_len:, 1]
        test_score = math.sqrt(mean_squared_error(
            test_target_data_plot, y_predicted_plot))
        scores.append(test_score)
        print('{}week Score: {} RMSE'.format(pred_len, test_score))
        df_output = pd.DataFrame({"predict": y_predicted_plot,
                                  "actual": test_target_data_plot})

        df_output.to_csv(
            out_dir+"df_seq2seq_{}_{}week.csv".format(region, pred_len))

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("{}:{}weeks".format(region, pred_len))
        ax.plot(y_predicted_plot, label="predict")
        ax.plot(test_target_data_plot, label="actual")
        ax.legend(loc="upper right")
        ax.grid(True)
        plt.show()

    return scores


if __name__ == "__main__":

    regions = ["New York", "Oregon", "Illinois",
               "California", "Texas", "Georgia"]
    score_dct = {}
    for region in regions:
        scores = main(region)
        score_dct[region] = scores
    score_df = pd.DataFrame(score_dct)
    score_df.to_csv("../output/scores_seq2seq.csv")
