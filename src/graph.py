import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def rmse_graph(region):
    df = pd.read_csv("../output/graph/bar_graph_rmse.csv")
    x_position = np.arange(4)
    y_base = df[region][df["Method"]=="sarima"][:4]
    y_lstm = df[region][df["Method"]=="simple_LSTM"][:4]
    y_seq2seq = df[region][df["Method"]=="seq2seq"][:4]
    y_atten = df[region][df["Method"]=="seq2seq_atten"][:4]
    fig = plt.figure(figsize=(12,5))
    plt.title(region)
    ax = fig.add_subplot(111)
    ax.bar(x_position,y_base,label="SARIMA",width=0.2)
    ax.bar(x_position+0.2,y_lstm,label="simple_LSTM",width=0.2)
    ax.bar(x_position+0.4,y_seq2seq,label="seq2seq",width=0.2)
    ax.bar(x_position+0.6,y_atten,label="seq2seq+attention",width=0.2)
    ax.set_xticks(x_position+0.2)
    ax.set_xticklabels([1,2,3,4,5])
    ax.set_xlabel("predict_week")
    ax.set_ylabel("RMSE")
    ax.legend(loc="upper left")
    plt.savefig(f"../output/RMSE_BarGraph_{region}.png")
    # plt.show()

def pearson_graph(region):
    df = pd.read_csv("../output/graph/bar_graph_pearson.csv")
    x_position = np.arange(4)
    y_base = df[region][df["Method"]=="sarima"][:4]
    y_lstm = df[region][df["Method"]=="simple_LSTM"][:4]
    y_seq2seq = df[region][df["Method"]=="seq2seq"][:4]
    y_atten = df[region][df["Method"]=="seq2seq_atten"][:4]
    fig = plt.figure(figsize=(12,5))
    plt.title(region)
    ax = fig.add_subplot(111)
    ax.bar(x_position,y_base,label="SARIMA",width=0.2)
    ax.bar(x_position+0.2,y_lstm,label="simple_LSTM",width=0.2)
    ax.bar(x_position+0.4,y_seq2seq,label="seq2seq",width=0.2)
    ax.bar(x_position+0.6,y_atten,label="seq2seq+attention",width=0.2)
    ax.set_xticks(x_position+0.2)
    ax.set_xticklabels([1,2,3,4,5])
    ax.set_xlabel("predict week")
    ax.set_ylabel("pearson correlation coefficient")
    ax.legend(loc="lower left")
    plt.savefig(f"../output/pearson_BarGraph_{region}.png")
    # plt.show()


    

if __name__ == "__main__":
    regions = ["New York", "Oregon","Illinois","California", "Texas", "Georgia"]
    for region in regions:
        rmse_graph(region)
        pearson_graph(region)