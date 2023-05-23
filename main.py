import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
import warnings
import sys
import os
from random import uniform
from sklearn import svm
from random import randint

data = pd.read_csv('iris.data', header=None)
features = data.iloc[:, :4].to_numpy()
labels = data.iloc[:, 4].to_numpy()
label_encoder = preprocessing.LabelEncoder()
target = label_encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=0)

#Linear Discriminant Analysis
#1
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=0)
lda = LinearDiscriminantAnalysis()
y_pred = lda.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
#2
print(f"Model score: {lda.fit(X_train, y_train).score(X_test, y_test) * 100}%")
#3
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=size)
    lda = LinearDiscriminantAnalysis()
    y_pred = lda.fit(X_train, y_train).predict(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(lda.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03, color="red", label="Classification accuracy")
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03, color="blue", label="Percentage of misclassified observations")
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('floralwhite')
plt.legend()
plt.show()
#4
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=size)
    lda = LinearDiscriminantAnalysis()
    y_pred = lda.fit(X_train, y_train).transform(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(lda.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03, color="red", label="Classification accuracy")
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03, color="blue", label="Percentage of misclassified observations")
ax.set_facecolor('seashell')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('floralwhite')
plt.legend()
plt.show()
#5
solver_parameters = ('svd', 'lsqr', 'eigen')
for parameter in solver_parameters:
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5)
    if parameter != 'svd':
        rand_shrinkage = uniform(0.05, 1)
        lda = LinearDiscriminantAnalysis(solver=parameter, shrinkage=rand_shrinkage)
        y_pred = lda.fit(X_train, y_train).predict(X_test)
        print(f'Solver: {parameter}, shrinkage: {rand_shrinkage} - number of mislabeled points out of a total {X_test.shape[0]} points: {(y_test != y_pred).sum()}')
    else:
        lda = LinearDiscriminantAnalysis(solver=parameter)
        y_pred = lda.fit(X_train, y_train).predict(X_test)
        print(f"Solver: {parameter}, number of mislabeled points out of a total {X_test.shape[0]} points: {(y_test != y_pred).sum()}")
#6
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5)
lda = LinearDiscriminantAnalysis(priors=[0.7, 0.15, 0.15])
y_pred = lda.fit(X_train, y_train).predict(X_test)
print(f"Number of mislabeled points out of a total {X_test.shape[0]} points : {(y_test != y_pred).sum()}")
print(f"Model score: {lda.fit(X_train, y_train).score(X_test, y_test) * 100}%")

# Support Vector Machines
# 1
clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of misclassified points: ", np.count_nonzero(y_test != y_pred))
print("Accuracy score: ", clf.score(features, labels))
# 2
print("Accuracy score: ", clf.fit(X_train, y_train).score(X_test, y_test))
# 3
print("Support vectors: ", clf.support_vectors_)
print("Support vector indices: ", clf.support_)
print("Number of support vectors for each class: ", clf.n_support_)
print('''"support_vectors_" - support vectors (largest linear classifier defined in characteristic space)
"support_" - indices of support vectors
"n_support_" - number of support vectors for each class.''')

# 4
size = 0
list_test_size = []
percentage_misclassified_observations = []
classification_accuracy = []
while size <= 0.95:
    size += 0.05
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=size)
    clf = svm.SVC()
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    list_test_size.append(size)
    percentage_misclassified_observations.append(np.count_nonzero(y_test != y_pred) / len(y_pred))
    classification_accuracy.append(clf.fit(X_train, y_train).score(X_test, y_test))
fig, ax = plt.subplots()
ax.bar(list_test_size, classification_accuracy, width=0.03)
ax.bar(list_test_size, percentage_misclassified_observations, width=0.03)
ax.set_facecolor('azure')
fig.set_figwidth(17)
fig.set_figheight(10)
fig.set_facecolor('mintcream')
plt.show()

# 5
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.05)
kernel_parameters = ('linear', 'poly', 'rbf', 'sigmoid')
for parameter in kernel_parameters:
    degree_rand = randint(2, 5)
    max_iter_random = randint(2, 5)
    clf = svm.SVC(kernel=parameter, degree=degree_rand, max_iter=max_iter_random)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    print(
        f'''With kernel={parameter}, degree={degree_rand}, max_iter={max_iter_random}, number of misclassified points: {np.count_nonzero(y_test != y_pred)}, accuracy score: {clf.score(features, labels)}, support vectors: {clf.support_vectors_}, support vector indices: {clf.support_}, number of support vectors for each class: {clf.n_support_}''')

# 6
clf = svm.NuSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(
    f'Misclassified points: {np.count_nonzero(y_test != y_pred)} Accuracy score: {clf.score(features, labels)}, support vectors: {clf.support_vectors_}, support vector indices: {clf.support_}, number of support vectors for each class: {clf.n_support_}')

clf = svm.LinearSVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(f'Misclassified points: {np.count_nonzero(y_test != y_pred)} Accuracy score: {clf.score(features, labels)}')
