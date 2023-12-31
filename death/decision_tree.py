from sklearn import tree
from death.data_set import (x_train, x_test, y_train, y_test, seed)
from library import gird_search, make_pickle

model_decision_tree = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed)
parameters = {
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
name = 'decision_tree'
path_pickle = f'death/pickle/{name}.pickle'
path_predict = f'death/pickle/predict/{name}.pickle'
gird_search(model_decision_tree, x_train, y_train, parameters, path_pickle)

make_pickle(path_pickle, path_predict, x_test, y_test)
