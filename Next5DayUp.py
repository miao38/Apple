import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import itertools

data = pd.read_csv("C:\\Users\\Roy Miao\\Documents\\Projects\\Apple\\Apple.csv")
data = data[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx",  "TodayVol", "SpyPx", "Next5DayUp"]]
data["Next5DayUp"] = data["Next5DayUp"].astype("int")
X = np.asanyarray(data[["FiveDayAvgPx", "FiveDayAvgVol", "TwoDayAvgPx", "TwoDayAvgVol", "TodayPx",  "TodayVol", "SpyPx"]])
X[0:5]
y = np.asanyarray(data["Next5DayUp"])
y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
#print("train set:", X_train.shape, y_train.shape)
#print("test setL:", X_train.shape, y_test.shape)
LR = LogisticRegression(C = .01, solver="liblinear").fit(X_train, y_train)
yHat = LR.predict(X_test)
yHat__prob = LR.predict_proba(X_test)
print(yHat__prob)

#confustion matrix part
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yHat, labels=[1,0]))

#compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yHat, labels=[1,0])
np.set_printoptions(precision=2)
#plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
print(classification_report(y_test, yHat))