from sklearn import linear_model
import pickle
from .data_set import (export_data, seed)
from main import gird_search

x_train, x_test, y_train, y_test = export_data()
model_logistic_regression = linear_model.LogisticRegression(class_weight='balanced', random_state=seed)
parameters = {
    'C': [0.01, 0.1, 1],
}
name = 'logistic_regression'
path_pickle = f'pickle/{name}.pickle'
gird_search(model_logistic_regression, x_train, y_train, parameters, path_pickle)

model_random_forest = pickle.load(open(path_pickle, 'rb'))
predict_random_forest = model_random_forest.predict(x_test)
with open(f'pickle/predict/{name}.pickle', 'wb') as f:
    pickle.dump(predict_random_forest, f)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])
