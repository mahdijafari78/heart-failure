from sklearn import ensemble
import pickle
from data_set import (x_train, x_test, y_train, y_test,seed)
from main import gird_search

model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=seed)
parameters = {
    "n_estimators": [5, 10, 15, 20],
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
name = 'random_forest'
path_pickle = f'pickle/{name}.pickle'
gird_search(model_random_forest, x_train, y_train, parameters, path_pickle)

model_random_forest = pickle.load(open(path_pickle, 'rb'))
predict_random_forest = model_random_forest.predict(x_test)
with open(f'pickle/predict/{name}.pickle', 'wb') as f:
    pickle.dump(predict_random_forest, f)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])
