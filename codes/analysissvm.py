import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV



datasetPos = pd.read_csv("positivoFR.csv")
datasetNeg = pd.read_csv("neg2lFR.csv")

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

param_grid = {'C': np.linspace(start = 0.001, stop = 1000, num = 20), 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'kernel': ['linear', 'sigmoid', 'rbf']}

svm = SVC()
print(svm.get_params())

svm_random = RandomizedSearchCV(estimator = svm, param_distributions = param_grid, cv = 5, verbose = 2, n_jobs = 4, n_iter = 50, scoring = 'f1')
svm_random.fit(x_train, y_train)
print('melhores ')
print(svm_random.best_params_)

params = svm_random.best_params_
#
# # **************************************************************
#
#
#
# # ******************************************colocando os melhores parametros na mao
# # params = {
# #     'n_estimators': 10,
# #     'min_samples_split': 2,
# #     'min_samples_leaf': 1,
# #     'max_features': None,
# #     'max_depth': 4,
# #     'criterion': 'gini',
# #     'class_weight': 'balanced',
# #     'bootstrap': True
# # }
# # ******************************************************************
#
#
svm = SVC(C = params['C'],
          kernel = params['kernel'],
          gamma = params['gamma'])

svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)
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
print ('ytest', y_test)
