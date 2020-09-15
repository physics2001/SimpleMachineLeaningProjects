import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

val_features = pd.read_csv('./val_features.csv')
val_labels = pd.read_csv('./val_labels.csv', header = None)

test_features = pd.read_csv('./test_features.csv')
test_labels = pd.read_csv('./test_labels.csv', header = None)

models = {}

for mdl in ['GB', 'RF', 'MLP']:
    models[mdl] = joblib.load('./{}_model.pkl'.format(mdl))
    print(models[mdl])


def evaluate_model(name, model, features, labels):
    start = time()
    pred = model.predict(features)
    end = time()
    accuracy = round(accuracy_score(labels, pred), 3)
    precision = round(precision_score(labels, pred, average='micro'), 3)
    recall = round(recall_score(labels, pred, average='micro'), 3)
    print('{} -- Accuracy: {} / Precision: {} / Recall: {} / Latency: {}ms'.format(name, accuracy, precision, recall, round((end - start), 5)))


for name, mdl in models.items():
    evaluate_model(name, mdl, val_features, val_labels)
    print(mdl)

evaluate_model('MLP', models['MLP'], test_features, test_labels)
