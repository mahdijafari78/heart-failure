from sklearn import neighbors
import pickle
from data_set import (x_train, x_test, y_train, y_test)
from main import gird_search

model_knn = neighbors.KNeighborsClassifier()
parameters = {
    "n_neighbors": list(range(1, 11)),
    "weights": ['uniform', 'distance'],
}
name = 'knn'
path_pickle = f'pickle/{name}.pickle'
gird_search(model_knn, x_train, y_train, parameters, path_pickle)
model_random_forest = pickle.load(open(path_pickle, 'rb'))
predict_random_forest = model_random_forest.predict(x_test)
with open(f'pickle/predict/{name}.pickle', 'wb') as f:
    pickle.dump(predict_random_forest, f)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])
