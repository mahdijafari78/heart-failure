from sklearn import svm
from death.data_set import (x_train, y_train, seed)
from library import gird_search

model_svm = svm.SVC(class_weight='balanced', random_state=seed)
parameters = {
    'C': [0.01, 0.1, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}
name = 'svm'
path_pickle = f'death/pickle/{name}.pickle'
gird_search(model_svm, x_train, y_train, parameters, path_pickle)
