import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv("C:\\Users\\roymi\\Documents\\Projects\\Apple\\Apple.csv")
predictors = data[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx",  "TodayVol", "SpyPx", "Next5DayPx"]]
viz = predictors[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx", "TodayVol", "SpyPx", "Next5DayPx"]]
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
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["FiveDayAvgPx"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print("5DayAvgPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #5.43
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #52.40
print("R2-score: %.2f" % r2_score(prediction, test_y))#.91

#using 5DayAvgVol
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["FiveDayAvgVol"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["FiveDayAvgVol"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("5DavAvgVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #19.79
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #19.79
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-184.98

#using TwoDayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TwoDayAvgPx"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TwoDayAvgPx"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("TwoDayAvgPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #5.07
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #44.68
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.92

#using TwoDayAvgPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TwoDayAvgVol"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TwoDayAvgVol"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("TwoDayAvgVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #19.87
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #615.58
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-237.72

#using TodayPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TodayPx"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TodayPx"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("TodayPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #4.38
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #36.25
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.94

#using TodayVol
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["TodayVol"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["TodayVol"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("TodayVol:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #19.71
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #610.25
print("R2-score: %.2f" % r2_score(prediction, test_y)) #-593.95

#using SpyPx
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["SpyPx"]])
train_y = np.asanyarray(train[["Next5DayPx"]])
regr.fit(train_x, train_y)
'''print(regr.coef_)
print(regr.intercept_)
plt.scatter(train.FiveDayAvgPx, train.NextDayPx)
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], "-r")
plt.xlabel("5DayAvgPx")
plt.ylabel("NextDayPx")
plt.show()'''

test_x = np.asanyarray(test[["SpyPx"]])
test_y = np.asanyarray(test[["Next5DayPx"]])
prediction = regr.predict(test_x)

print()
print("SpyPx:")
print("MAE: %.2f" % np.mean(np.absolute((prediction - test_y)))) #8.54
print("MSE: %.2f" % np.mean((prediction - test_y) ** 2)) #140.42
print("R2-score: %.2f" % r2_score(prediction, test_y)) #.72