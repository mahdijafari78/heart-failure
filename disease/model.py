from sklearn import tree
from death.data_set import (x_train, y_train, seed)
from library import gird_search
from sklearn import neighbors, svm, linear_model,model_selection
from sklearn import ensemble
import lightgbm

base_name = 'disease'


def ada_boost_model():
    model_decision_tree = tree.DecisionTreeClassifier(class_weight='balanced',random_state=seed,
                                                      max_depth=None)
    model_ada_boost = ensemble.AdaBoostClassifier(model_decision_tree)
    parameters = {
        "base_estimator__criterion": ["gini", "entropy"],
        "base_estimator__splitter": ["best", "random"],
        "n_estimators": [1, 2],
    }
    name = 'ada_boost'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_ada_boost, x_train, y_train, parameters, path_pickle, scoring='f1')


def decision_tree_model():
    model_decision_tree = tree.DecisionTreeClassifier(class_weight='balanced', random_state=seed)
    parameters = {
        "max_depth": [1, 2, 3, 5, 10, None],
        "min_samples_leaf": [1, 5, 10, 20]
    }
    name = 'decision_tree'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_decision_tree, x_train, y_train, parameters, path_pickle)


def knn_model():
    model_knn = neighbors.KNeighborsClassifier()
    parameters = {
        "n_neighbors": list(range(1, 11)),
        "weights": ['uniform', 'distance'],
    }
    name = 'knn'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_knn, x_train, y_train, parameters, path_pickle)


def random_forest_model():
    model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=seed)
    parameters = {
        "n_estimators": [5, 10, 15, 20],
        "max_depth": [1, 2, 3, 5, 10, None],
        "min_samples_leaf": [1, 5, 10, 20]
    }
    name = 'random_forest'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_random_forest, x_train, y_train, parameters, path_pickle)


def svm_model():
    model_svm = svm.SVC(class_weight='balanced', random_state=seed)
    parameters = {
        'C': [0.01, 0.1, 1],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
    }
    name = 'svm'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_svm, x_train, y_train, parameters, path_pickle)


def light_gbm_model():
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
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_lightgbm, x_train, y_train, parameters, path_pickle)


def logistic_regression_model():
    model_logistic_regression = linear_model.LogisticRegression(class_weight='balanced', random_state=seed)
    parameters = {
        'C': [0.01, 0.1, 1],
    }
    name = 'logistic_regression'
    path_pickle = f'{base_name}/pickle/{name}.pickle'
    print(name)
    gird_search(model_logistic_regression, x_train, y_train, parameters, path_pickle)
