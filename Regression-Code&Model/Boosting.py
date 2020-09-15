import numpy as np
import pandas as pd
import joblib
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import seaborn as sns

warnings.filterwarnings('ignore', category =FutureWarning)
warnings.filterwarnings('ignore', category =DeprecationWarning)

tr_features = pd.read_csv('./train_features.csv')
tr_labels = pd.read_csv('./train_labels.csv', header = None)

print(tr_features.head())
print(tr_labels.head())


def print_results(results):
    print('Best Hyperparams: {}\n'.format(results.best_params_))

    score = -results.cv_results_['mean_test_score']
    for mse, params in zip(score, results.cv_results_['params']):
        print('mse: {} / for {}'.format(round(mse, 6), params))


def print_regression_results(mdl):
    print('Coefficients: \n', mdl.coef_)

    pred_labels =  mdl.predict(tr_features)
    print('Mean squared error: %.2f'
          % mean_squared_error(tr_labels, pred_labels))

    print('Coefficient of determination: %.2f'
          % r2_score(tr_labels, pred_labels))


gb = GradientBoostingRegressor()

parameters = {
    'n_estimators': [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80],
    'max_depth':[1]
}

cv = GridSearchCV(gb, parameters, cv=5, scoring='neg_mean_squared_error')
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)

joblib.dump(cv.best_estimator_, './GB.pkl')