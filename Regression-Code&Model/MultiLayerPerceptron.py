import numpy as np
import pandas as pd
import joblib
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import warnings
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, SCORERS

warnings.filterwarnings('ignore', category =FutureWarning)
warnings.filterwarnings('ignore', category =DeprecationWarning)

tr_features = pd.read_csv('./train_features.csv')
tr_labels = pd.read_csv('./train_labels.csv', header = None)

print(tr_features.head())
print(tr_labels.head())
print(SCORERS.keys())


def print_results(results):
    print('Best Hyperparams: {}\n'.format(results.best_params_))

    score = -results.cv_results_['mean_test_score']
    for mse, params in zip(score, results.cv_results_['params']):
        print('mse: {} / for {}'.format(round(mse, 6), params))


mlp = MLPRegressor(max_iter=1000)

parameters = {
    'hidden_layer_sizes': [(200, 200), (200, 200, 200), (100, 100, 100)],
    'activation': ['identity'],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

cv = GridSearchCV(mlp, parameters, cv=5, scoring='neg_mean_squared_error')
cv.fit(tr_features, tr_labels.values.ravel())
print(cv.cv_results_)
print_results(cv)

joblib.dump(cv.best_estimator_, './MLP.pkl')