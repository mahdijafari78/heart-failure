import lightgbm
from death.data_set import (x_train, x_test, y_train, y_test, seed)
from library import gird_search, make_pickle

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
path_pickle = f'death/pickle/{name}.pickle'
path_predict = f'death/pickle/predict/{name}.pickle'
gird_search(model_lightgbm, x_train, y_train, parameters, path_pickle)
make_pickle(path_pickle, path_predict, x_test, y_test)
