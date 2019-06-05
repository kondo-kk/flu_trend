from train import train
from predict_return_result import predict
import matplotlib.pyplot as plt
import pandas as pd

in_dir = "../src/"
out_dir = "../output/"

def plot(predicted, actual,region,p):
    
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.plot(actual,label="actual", c="orange")
    ax.plot(predicted,label="predicted", c="royalblue")
    ax.legend()
    ax.grid(True)
    plt.savefig(out_dir+f"{region}_plot_p{p}.png")
    # plt.show()

def to_csv(predicted, actual,region,p):
    df_plot = pd.DataFrame({"predict":predicted,
                             "actual":actual})
    df_plot.to_csv(out_dir+f"df_plot_{region}_p{p}.csv")




if __name__ == "__main__":
    regions = ["New York", "oregon","Illinois","California", "Texas", "Georgia"]
    
    
    for region in regions:
        score_dct={"rmse":[],
               "r":[]}
        for p in range(5):
            predicted, actual, rmse, r = predict(region,p)
            plot(predicted, actual, region, p)
            to_csv(predicted, actual, region, p)
            score_dct["rmse"].append(rmse)
            score_dct["r"].append(r)
        df_score = pd.DataFrame(score_dct)
        df_score.to_csv(out_dir+f"df_score_{region}.csv")


