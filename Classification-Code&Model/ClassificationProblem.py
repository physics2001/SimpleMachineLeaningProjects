import numpy as np
import pandas as pd 
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns


data = datasets.load_iris()


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df


df_data = sklearn_to_df(data)

print(df_data.head())

print(df_data.describe())

features = df_data.drop('target', axis = 1)
labels = df_data['target']

Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.4)
Xval, Xtest, yval, ytest = train_test_split(Xtest, ytest, test_size=0.5)

print(Xtrain.describe())
print(Xval.describe())
print(Xtest.describe())
print(ytrain.describe())
print(yval.describe())
print(ytest.describe())

Xtrain.to_csv('./train_features.csv', index = False)
Xval.to_csv('./val_features.csv', index = False)
Xtest.to_csv('./test_features.csv', index = False)

ytrain.to_csv('./train_labels.csv', index = False)
yval.to_csv('./val_labels.csv', index = False)
ytest.to_csv('./test_labels.csv', index = False)