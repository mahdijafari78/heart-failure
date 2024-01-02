from sklearn import tree
from death.data_set import (x_train, y_train, seed)
from library import gird_search

model_decision_tree = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed)
parameters = {
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
name = 'decision_tree'
path_pickle = f'death/pickle/{name}.pickle'
gird_search(model_decision_tree, x_train, y_train, parameters, path_pickle)

