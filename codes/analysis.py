import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.model_selection import RandomizedSearchCV



datasetPos = pd.read_csv("positivoFC.csv")
datasetNeg = pd.read_csv("neg1lFC.csv")

sizePos = len(datasetPos.index)
print(sizePos)

datasetFull = pd.concat([datasetPos, datasetNeg])
print(len(datasetFull.index))
print(datasetFull)
datasetFullDropped = datasetFull.drop(["nameseq", "label"], axis=1)
print(datasetFullDropped)

y = np.zeros(len(datasetFull.index))

for i in range(sizePos):
    y[i] = 1

print(y)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 5000, num = 10)]
# Number of features to consider at every split
max_features = [None, 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

criterion = ['gini', 'entropy', 'log_loss']

class_weight = ['balanced', 'balanced_subsample', None]

param_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf,
    'bootstrap': bootstrap,
    'criterion': criterion,
    'class_weight': class_weight
}

print(param_grid)


x_train, x_test, y_train, y_test = train_test_split(datasetFullDropped, y, test_size=0.20, random_state=97, stratify=y)
# print(x_train)

scaler = Normalizer().fit(x_train)
scaler = Normalizer().fit(x_test)

scaler1 = Normalizer().fit(x_train)
scaler2 = Normalizer().fit(x_test)
x_train = scaler1.transform(x_train)
x_test = scaler2.transform(x_test)


x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train)

# *************para testar os parametros*****************************************

# Pegando os melhores parametros
rf = RandomForestClassifier(random_state=97)
print(rf.get_params())

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid, cv = 5, verbose = 2, n_jobs = 4, n_iter = 10, scoring = 'f1')

rf_random.fit(x_train, y_train)
print('melhores ')
print(rf_random.best_params_)

params = rf_random.best_params_

# **************************************************************



# ******************************************colocando os melhores parametros na mao
# params = {
#     'n_estimators': 10,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
#     'max_features': None,
#     'max_depth': 4,
#     'criterion': 'gini',
#     'class_weight': 'balanced',
#     'bootstrap': True
# }
# ******************************************************************


rf = RandomForestClassifier(
    random_state = 97,
    n_estimators = params["n_estimators"],
    max_features = params["max_features"],
    max_depth = params["max_depth"],
    min_samples_split = params["min_samples_split"],
    min_samples_leaf = params["min_samples_leaf"],
    bootstrap = params["bootstrap"]
)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
#

#
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

#
#
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision score: {precision_score(y_test, y_pred, average = None)}")
print(f"Recall score: {recall_score(y_test, y_pred, average = None)}")
print(f"AUC: {auc(fpr, tpr)}")
print(f"Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))
print(f"F1: {f1_score(y_test, y_pred)}")
