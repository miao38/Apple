import pandas as pd
import numpy as np

dataset = pd.read_csv("C:\\Users\\Roy Miao\\Documents\\Projects\\Apple\\Original Apple.csv")
dataset = dataset[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

#5 day Px
adj_close = np.asanyarray(dataset["Adj Close"])
five_day_Px_list = []
five_Px = 0
avg_five_Px = 0
for data in range(len(adj_close) - 1):
    if data < 5:
        five_Px += float(adj_close[data])
        if data == 4:
            avg_five_Px = five_Px / 5
            five_day_Px_list.append(avg_five_Px)
    else:
        five_Px = five_Px + adj_close[data] - adj_close[data - 5]
        avg_five_Px = five_Px / 5
        five_day_Px_list.append(avg_five_Px)

#5 day vol
volume = np.asanyarray(dataset["Volume"])
five_day_vol = []
five_vol = 0
avg_five_vol = 0
for data in range(len(volume) - 1):
    if data < 5:
        five_vol += float(volume[data])
        if data == 4:
            avg_five_vol = five_vol / 5
            five_day_vol.append(avg_five_vol)
    else:
        five_vol = five_vol + volume[data] - volume[data - 5]
        avg_five_vol = five_vol / 5
        five_day_vol.append(avg_five_vol)

#2 day Px
two_day_Px = []
two_Px = 0
avg_two_Px = 0
for data in range(len(adj_close) - 1):
    if data == 0 or data == 1 or data == 2:
        continue
    elif data < 5:
        two_Px += float(adj_close[data])
        if data == 4:
            avg_two_Px = two_Px / 2
            two_day_Px.append(avg_two_Px)
    else:
        two_Px = two_Px + adj_close[data] - adj_close[data - 2]
        avg_two_Px = two_Px / 2
        two_day_Px.append(avg_two_Px)

#2 day vol
two_day_vol = []
two_vol = 0
avg_two_vol = 0
for data in range(len(volume) - 1):
    if data == 0 or data == 1 or data == 2:
        continue
    elif data < 5:
        two_vol += float(volume[data])
        if data == 4:
            avg_two_vol = two_vol / 2
            two_day_vol.append(avg_two_vol)
    else:
        two_vol = two_vol + volume[data] - volume[data - 2]
        avg_two_vol = two_vol / 2
        two_day_vol.append(avg_two_vol)

#Day Px
day_Px = []
place = 0
for data in adj_close:
    if place > 4:
        day_Px.append(data)
    place += 1

#Day vol
day_vol = []
place = 0
for data in volume:
    if place > 4:
        day_vol.append(data)
    place += 1

#Next Day Px
next_day_Px = day_Px.copy()
del next_day_Px[0]
next_day_Px.append(0)

#Next 5 Day Px
next_5_day_Px = []
five = 1
place = 0
for i in range(len(next_day_Px)):
    next_5_day_Px.append(0)
for day in range(len(next_day_Px)):
    if five == 5:
        next_5_day_Px[place] = next_day_Px[day]
        place += 1
    elif five > 5:
        next_5_day_Px[place] = next_day_Px[day]
        place += 1
    five += 1

#Next 5 Day Up
Up = []
for day in range(len(next_day_Px)):
    if(next_5_day_Px[day] > day_Px[day]):
        Up.append(1)
    else:
        Up.append(0)

#saving to csv
submission = pd.DataFrame({"5 Day Px": five_day_Px_list, "5 Day Vol": five_day_vol, "2 Day Px": two_day_Px, "2 Day Vol": two_day_vol,
                           "Day Px": day_Px, "Day Vol": day_vol, "Next Day Px": next_day_Px, "Next 5 Day Px": next_5_day_Px, "Next 5 Day Up": Up})
submission.to_csv("C:\\Users\\Roy Miao\\Documents\\Projects\\Apple\\Submission.csv", index = False)