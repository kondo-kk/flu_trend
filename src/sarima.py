import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

in_dir = "../src/"
out_dir = "../output/"

def calc_rmse(pred, target):
    return np.sqrt(np.mean((target - pred)**2))

def calc_pearson(pred,target):
    return pearsonr(target.ravel(), pred.ravel())[0]

#他の手法ではtest_target_dataはt=300から始まるのでそれに合わせて300から予測開始するように訓練データを作成
#126 = len(test_data) - input_len - predict_len -1
def sarima_dataset(dataset,input_len=104,pred_len=5):
    target_data = [dataset[300+i:300+i+pred_len,1] for i in range(126)]
    train_data = [np.array([dataset[298-input_len+i:298+i,0], dataset[299-input_len+i:299+i,1]]).T for i in range(126)] 
    target_ex_data =  np.array([dataset[299+i:299+i+pred_len,0] for i in range(126)]).reshape(126,pred_len,1)
    return np.array(train_data), np.array(target_data), target_ex_data


def sarima_train(region):
    input_len = 208
    predict_len = 5

    df=pd.read_csv(in_dir+"df_{}_2010-2018.csv".format(region),index_col=0)
    dataset = df.values
    dataset=dataset.astype("float32")
    train_size=int(len(dataset)*0.67)
    train_ds=dataset[0:train_size,:]
    test_ds=dataset[train_size:len(dataset),:]
    train_max=np.amax(train_ds,axis=0)[1]
    train_min=np.amin(train_ds,axis=0)[1]
    scaler=MinMaxScaler(feature_range=(0,1))
    scaler_train=scaler.fit(train_ds)
    dataset_scaled=scaler_train.transform(dataset)

    train_data, target_data, target_ex_data = sarima_dataset(dataset, input_len=input_len,pred_len=predict_len)
    
    predicted = []
    for epoch in range(len(train_data)):
        model = sm.tsa.statespace.SARIMAX(endog =train_data[epoch,:,1],
                                # exog = train_data[epoch,:,0],
                                order = (3,0,2),
                                seasonal_order = (1,0,0,52),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
        # predicted.append(model.forecast(steps = predict_len, exog=target_ex_data[epoch]))
        predicted.append(model.forecast(steps = predict_len))
    predicted = np.array(predicted)

    rmse = [calc_rmse(predicted[:,p],target_data[:,p]) for p in range(predict_len)]
    r = [calc_pearson(predicted[:,p],target_data[:,p]) for p in range(predict_len)]
    print("rmse:",rmse)
    print("r:",r)

    return predicted, target_data, rmse, r 
    
def plot(predicted, target, region, p):
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    ax.plot(target[:,p],label="actual", c="orange")
    ax.plot(predicted[:,p],label="predicted", c="royalblue")
    ax.legend()
    ax.grid(True)
    plt.savefig(out_dir+f"{region}_sarima_p{p}.png")
    # plt.show()

def to_csv(predicted, target, region, p):
    df_plot = pd.DataFrame({"predict":predicted[:,p],
                             "actual":target[:,p]})
    df_plot.to_csv(out_dir+f"df_sarima_{region}_p{p}.csv")


if __name__ == "__main__":
    regions=["New York", "oregon","Illinois","California", "Texas", "georgia"]

    rmse_dct = {}
    r_dct = {}
    for region in regions:
        predicted, target, rmse, r = sarima_train(region)
        rmse_dct[region] = rmse
        r_dct[region] = r
        for p in range(5):
            plot(predicted, target, region, p)
            to_csv(predicted, target, region, p)
    pd.DataFrame(rmse_dct).to_csv(out_dir+f"sarima_rmse.csv")
    pd.DataFrame(r_dct).to_csv(out_dir+f"sarima_pearson.csv")
        
    
        