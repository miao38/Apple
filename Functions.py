import pandas as pd
import numpy as np

def Px (days):
    adj_close = np.asanyarray(dataset["Adj Close"])
    Px_list = []
    Px = 0
    avg_Px = 0
    for data in range(len(adj_close) - 1):
        if data < :
            Px += float(adj_close[data])
            if data == 4:
                avg_Px = Px / days
                Px_list.append(avg_Px)
        else:
            Px = Px + adj_close[data] - adj_close[data - days]
            avg_Px = Px / days
            Px_list.append(avg_Px)

def vol (days):
    volume = np.asanyarray(dataset["Volume"])
    vol_list = []
    vol = 0
    avg_vol = 0
    for data in range(len(volume) - 1):
        if data < 5:
            vol += float(volume[data])
            if data == 4:
                avg_vol = vol / 5
                vol_list.append(avg_vol)
        else:
            vol = vol + volume[data] - volume[data - 5]
            avg_vol = vol / 5
            vol_list.append(avg_vol)


dataset = pd.read_csv("C:\\Users\\Roy Miao\\Documents\\Projects\\Apple\\Original Apple.csv")
dataset = dataset[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]