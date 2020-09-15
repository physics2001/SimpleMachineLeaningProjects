import numpy as np
import pandas as pd
import joblib
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

tr_features = pd.read_csv('./train_features.csv')
tr_labels = pd.read_csv('./train_labels.csv', header=None)

tr_features.drop(['sex'], axis=1, inplace=True)

print(tr_features.head())
print(tr_labels.head())

lr = LinearRegression()

mdl = lr.fit(tr_features, tr_labels)

scores = cross_val_score(lr, tr_features, tr_labels, cv=5, scoring='neg_mean_squared_error')
print(scores)

mse_scores = -scores
print(mse_scores)

rmse_scores = np.sqrt(mse_scores)
print(rmse_scores)

print(rmse_scores.mean())

def print_regression_results(mdl):
    print('Coefficients: \n', mdl.coef_)

    pred_labels =  mdl.predict(tr_features)
    print('Mean squared error: %.2f'
          % mean_squared_error(tr_labels, pred_labels))

    print('Coefficient of determination: %.2f'
          % r2_score(tr_labels, pred_labels))


print_regression_results(mdl)

joblib.dump(mdl, './LRwithoutSex.pkl')
