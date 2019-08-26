import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("C:\\Users\\roymi\\Documents\\Projects\\Apple\\Apple.csv")
predictors = data[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx",  "TodayVol", "SpyPx", "NextDayPx"]]
viz = predictors[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx", "TodayVol", "SpyPx", "NextDayPx"]]
#viz.hist()
#plt.show()

x = np.random.rand(len(data)) < .7
train = predictors[x]
test = predictors[~x]
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx,)
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show() '''

#using 5DayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["FiveDayAvgPx"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["FiveDayAvgPx"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print("5DayAvgPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #3.37
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #21.7
print("R2-score: %.2f" % r2_score(prediction, test_y))#.97

#using 5DayAvgVol
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["FiveDayAvgVol"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["FiveDayAvgVol"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("5DavAvgVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #23.65
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #814.5
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-92.29

#using TwoDayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TwoDayAvgPx"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TwoDayAvgPx"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("TwoDayAvgPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #2.46
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #13.19
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.98

#using TwoDayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TwoDayAvgVol"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TwoDayAvgVol"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("TwoDayAvgVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #21.61
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #704.5
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-307.25

#using TodayPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TodayPx"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TodayPx"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("TodayPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #1.42
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #3.96
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.99

#using TodayVol
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TodayVol"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TodayVol"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("TodayVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #21.43
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #737.07
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-9.9084.96

#using SpyPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["SpyPx"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["SpyPx"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("SpyPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #8.75
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #136.72
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.76

#TodayPx, TwoDayAvgPx, FiveDayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TodayPx", "TwoDayAvgPx", "FiveDayAvgPx"]])
train_y = np.asanyarray(train[["NextDayPx"]])
regr.fit(train_x, train_y)
print(regr.coef_)
print(regr.intercept_)
'''plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["SpyPx", "TwoDayAvgPx", "FiveDayAvgPx"]])
test_y = np.asanyarray(test[["NextDayPx"]])
prediction = regr.predict(test_x)

print()
print("TodayPx, TwoDayAvgPx, FiveDayAvgPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #79.64
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #6452.51
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-14.11