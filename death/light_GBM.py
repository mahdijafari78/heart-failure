import lightgbm
import pickle
from .data_set import (x_train, x_test, y_train, y_test, seed)
from main import gird_search

model_lightgbm = lightgbm.LGBMClassifier(class_weight='balanced', random_state=seed)
parameters = {
    'num_leaves': [7, 15, 31],
    'learning_rate': [0.001, 0.01, 0.1],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [1],
    'reg_lambda': [1],
    'colsample_bytree': [0.5, 0.75, 1.]
}
name = 'lightgbm'
path_pickle = f'pickle/{name}.pickle'
gird_search(model_lightgbm, x_train, y_train, parameters, path_pickle)

model_random_forest = pickle.load(open(path_pickle, 'rb'))
predict_random_forest = model_random_forest.predict(x_test)
with open(f'pickle/predict/{name}.pickle', 'wb') as f:
    pickle.dump(predict_random_forest, f)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])
