from sklearn import ensemble
from death.data_set import (x_train, x_test, y_train, y_test, seed)
from library import gird_search, make_pickle

model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=seed)
parameters = {
    "n_estimators": [5, 10, 15, 20],
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
name = 'random_forest'
path_pickle = f'death/pickle/{name}.pickle'
path_predict = f'death/pickle/predict/{name}.pickle'
gird_search(model_random_forest, x_train, y_train, parameters, path_pickle)

make_pickle(path_pickle, path_predict, x_test, y_test)
