import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from time import time

val_features = pd.read_csv('./val_features.csv')
val_labels = pd.read_csv('./val_labels.csv', header = None)

test_features = pd.read_csv('./test_features.csv')
test_labels = pd.read_csv('./test_labels.csv', header = None)

models = {}

for mdl in ['LR', 'GB', 'MLP']:
    models[mdl] = joblib.load('./{}.pkl'.format(mdl))
    print(models[mdl])


def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    rmse = round(np.sqrt(mean_squared_error(labels, pred)), 6)
    r2 = round(r2_score(labels, pred), 6)
    print('{} -- rmse: {} / r2: {} / Latency: {}ms'.format(name, rmse, r2, round((end - start), 5)))


for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)
    print(mdl)

evaluate_model('MLP', models['MLP'], test_features, test_labels)
evaluate_model('LR', models['LR'], test_features, test_labels)