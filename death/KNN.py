from sklearn import neighbors
from death.data_set import (x_train, x_test, y_train, y_test)
from library import gird_search, make_pickle

model_knn = neighbors.KNeighborsClassifier()
parameters = {
    "n_neighbors": list(range(1, 11)),
    "weights": ['uniform', 'distance'],
}
name = 'knn'
path_pickle = f'death/pickle/{name}.pickle'
path_predict = f'death/pickle/predict/{name}.pickle'
gird_search(model_knn, x_train, y_train, parameters, path_pickle)
make_pickle(path_pickle, path_predict, x_test, y_test)
