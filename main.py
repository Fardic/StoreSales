from datetime import date
from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
from torch.nn import RNN
from sklearn.model_selection import train_test_split
from dataset import rnn_custom_dataset
from torch.utils.data import DataLoader

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    


    sequence_length = 64
    num_layers = 4
    hidden_size = 512
    num_epochs = 5
    batch_size = 64
    learning_rate = 0.001

    money = prepare_for_rnn(np.array(oil.loc[oil["dcoilwtico"].notnull()]), sequence_length)
    money_x_train, money_x_valid, money_y_train, money_y_valid = train_test_split(money[:, :, 2], money[:, :, 1], 
                                                                                 test_size=0.1, random_state=7)
                        

    train_money = rnn_custom_dataset(np.stack([money_x_train, money_y_train], axis=2))
    valid_money = rnn_custom_dataset(np.stack([money_x_valid, money_y_valid], axis=2))

    dataloader_train = DataLoader(dataset=train_money, batch_size=batch_size, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(dataset=valid_money, batch_size=len(valid_money), shuffle=False, num_workers=1)


    class RNN(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.num_layers = num_layers
            self.hidden_size = hidden_size

            self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = torch.nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device=device)
            out, _ = self.rnn(x, h0)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    model_money = RNN(1, hidden_size, num_layers).to(device=device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model_money.parameters(), lr = learning_rate)


    n_total_steps = len(dataloader_train)
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader_train):  
            features = features.to(device)
            label = labels[:, -1].to(device)
            
            # Forward pass
            outputs = model_money(features)
            loss = criterion(outputs, label.view(-1, 1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
    
    with torch.no_grad():
        for features, labels in dataloader_val:
            features = features.to(device)
            label = labels[:, -1].to(device)
            outputs = model_money(features)
            loss = criterion(outputs, label.view(-1, 1))

        print(f'Loss of the network : {loss.item():.4f} ')





























    # # Oil prices are added to the dataset
    # train = train.merge(oil, right_on="date", left_on="date")
    # test = test.merge(oil, right_on="date", left_on="date")



    # # print(train.shape)
    # # print(oil[oil["dcoilwtico"].isnull()])
    # # print(train.head())





