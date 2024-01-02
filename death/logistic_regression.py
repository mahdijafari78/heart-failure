from sklearn import linear_model
from death.data_set import (x_train, y_train, seed)
from library import gird_search

model_logistic_regression = linear_model.LogisticRegression(class_weight='balanced', random_state=seed)
parameters = {
    'C': [0.01, 0.1, 1],
}
name = 'logistic_regression'
path_pickle = f'death/pickle/{name}.pickle'
gird_search(model_logistic_regression, x_train, y_train, parameters, path_pickle)
