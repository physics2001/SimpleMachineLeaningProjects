import numpy as np
import pandas as pd
import joblib
import statsmodels as statsmodels
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
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

    means = results.cv_results_['mean_test_score']
    sds = results.cv_results_['std_test_score']
    for mean, sd, params in zip(means, sds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(sd*2, 3), params))


gb = GradientBoostingClassifier()

parameters = {
    'n_estimators':[50, 55, 60],
    'max_depth':[2]
}

cv = GridSearchCV(gb, parameters, cv=5)
cv.fit(tr_features, tr_labels.values.ravel())

print_results(cv)

joblib.dump(cv.best_estimator_, './GB_model.pkl')