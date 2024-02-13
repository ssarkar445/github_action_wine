import os
import pandas as pd
import cfg
from metrics import ClassificationMetrics
from sklearn import preprocessing
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def train_model(model,df,fold,metric):

    train_df = df.query("kfold!=@fold").reset_index(drop=True)
    valid_df = df.query("kfold==@fold").reset_index(drop=True)


    xtrain = train_df[cfg.FEATURES]
    xvalid = valid_df[cfg.FEATURES]

    ytrain = train_df[cfg.TARGET]
    yvalid = valid_df[cfg.TARGET]

    idtrain = train_df[cfg.IDENTIFIER]
    idvalid = valid_df[cfg.IDENTIFIER]

    scaler = preprocessing.StandardScaler()

    xtrain[cfg.FEATURES] = scaler.fit_transform(xtrain[cfg.FEATURES])
    xvalid[cfg.FEATURES] = scaler.transform(xvalid[cfg.FEATURES])


    model.fit(xtrain,ytrain)

    pred = model.predict(xvalid)

    print(f"For fold {fold} accuracy={metric._accuracy(yvalid,pred)}")

    return model,scaler


if __name__ == "__main__":
    df = pd.read_csv(cfg.TRAINING_DATA)
    metric = ClassificationMetrics()
    model = cfg.MODEL

    for fold in range(0,5):
        model,scaler = train_model(model,df,fold,metric)
        joblib.dump(scaler,open(Path(f"../scaler_pkl/{fold}_scaler.pkl"),'wb'))
        joblib.dump(model,open(Path(f"../model_pkl/{fold}.pkl"),'wb'))