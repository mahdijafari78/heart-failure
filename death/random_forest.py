from sklearn import ensemble
from death.data_set import (x_train, y_train, seed)
from library import gird_search

model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=seed)
parameters = {
    "n_estimators": [5, 10, 15, 20],
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
name = 'random_forest'
path_pickle = f'death/pickle/{name}.pickle'
gird_search(model_random_forest, x_train, y_train, parameters, path_pickle)

