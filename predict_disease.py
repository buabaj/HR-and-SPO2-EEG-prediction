import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
os.chdir("C:/Users/eyadk/Desktop/dela/")
dataset = pd.read_excel("full_merged_data.xlsx")


def change_gender_label(gender):
    if gender=="M":
        return 0
    else:
        return 1

dataset["Gender"] = dataset.apply(lambda x: change_gender_label(x["Gender"]),axis=1)
########## Machine Learning Model
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

dataset = dataset.drop(["Subject_ID"],axis=1)
exclude_cols = ['label']


X = dataset.filter(regex="^(?!({0})$).*$".format('|'.join(exclude_cols)))

Y = dataset["label"]

# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
transformer = scaler.fit(X)

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison AUC')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()




