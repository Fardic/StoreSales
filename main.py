from datetime import date
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import tensorflow as tf

if __name__=="__main__":

    train = pd.read_csv("data/train.csv")
    train["onpromotion"] = train["onpromotion"].astype(int)
    train["date"] = pd.to_datetime(train["date"])

    test = pd.read_csv("data/test.csv")
    test["onpromotion"] = test["onpromotion"].astype(int)
    test["date"] = pd.to_datetime(test["date"])


    transactions = pd.read_csv("data/transactions.csv")


    # Store clusters are inserted to the dataset
    stores = pd.read_csv("data/stores.csv")
    train = train.merge(stores[["store_nbr", "cluster"]], right_on="store_nbr", left_on="store_nbr")
    test = test.merge(stores[["store_nbr", "cluster"]], right_on="store_nbr", left_on="store_nbr")


    oil = pd.read_csv("data/oil.csv")
    oil["date"] = pd.to_datetime(oil["date"])


    # Outline observation of oil prices 
    # plt.plot(oil["date"], oil["dcoilwtico"])
    # plt.show()

    oil["date_id"] = np.arange(start=0, stop=oil.shape[0])

    def prepare_for_rnn(data, length):
        rnn_data = []
        for i in range(length, data.shape[0]):
            rnn_data.append(data[i-length:i])
        return np.array(rnn_data)


    # a = prepare_for_rnn(np.array(oil), 3)
    # print(a.shape)
    # simple_rnn = tf.keras.layers.SimpleRNN(1, activation="relu", dropout=0.2)
    # output = simple_rnn(np.array(oil.loc[oil["dcoilwtico"].notnull(), "date_id"]).reshape(-1, 1))


    polyreg=make_pipeline(PolynomialFeatures(4),LinearRegression())
    polyreg.fit(np.array(oil.loc[oil["dcoilwtico"].notnull(), "date_id"]).reshape(-1, 1), oil.loc[oil["dcoilwtico"].notnull(), "dcoilwtico"])

    plt.plot(oil["date"], oil["dcoilwtico"])
    plt.plot(oil["date"], polyreg.predict(np.array(oil["date_id"]).reshape(-1, 1)))
    plt.show()


    # Oil prices are added to the dataset
    train = train.merge(oil, right_on="date", left_on="date")
    test = test.merge(oil, right_on="date", left_on="date")



    # print(train.shape)
    # print(oil[oil["dcoilwtico"].isnull()])
    # print(train.head())





