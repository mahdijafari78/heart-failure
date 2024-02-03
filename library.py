import datetime
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics, preprocessing
import numpy as np
import json


class StartDataset:
    package = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    seed = 42

    def __init__(self, file, n=True):
        self.file = file
        self.n = n

    def get_package(self):
        path_json = os.path.join(os.getcwd(), 'package.json')
        with open(path_json, 'r') as file_json:
            package = json.load(file_json)
        if package.get(self.file) is None:
            raise 'there is no such config'
        self.package = package.get(self.file)

    def get_dataset(self):
        if self.package is not None:
            self.seed = self.package.get('random_state', self.seed)
            path = os.path.join(os.getcwd(), self.package.get('path_dataset'))
            data = pd.read_csv(path)
            data_x = np.array(data.drop(columns=self.package.get('data_x_drop')))
            data_y = np.array(data[self.package.get('data_y')])
            self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(data_x, data_y,
                                                                                                    test_size=self.package.get(
                                                                                                        'test_size',
                                                                                                        0.2),
                                                                                                    random_state=self.seed,
                                                                                                    shuffle=self.package.get(
                                                                                                        'shuffle',
                                                                                                        True),
                                                                                                    stratify=data_y)
            if self.n:
                self.normal()
        else:
            print('pls run function get_package')

    def normal(self):
        if self.package is not None:
            self.x_train, self.x_test = normal_data(self.x_train, self.x_test, self.package.get('normal_data'))
        else:
            print('pls run function get_package')


def gird_search(model, x_train, y_train, parameters, path, scoring='f1'):
    model = model_selection.GridSearchCV(model, parameters, scoring=scoring)
    name_base = path.split('/')[0]
    path = os.path.join(os.getcwd(), path)
    model.fit(x_train, y_train)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    beat_path = os.path.join(os.getcwd(), 'output')
    with open(f'{beat_path}/{name_base}-best-params.txt', 'a') as file:
        file.write(f'{os.path.basename(path).split(".")[0]}:{model.best_params_}\n')
    print(f'best_params_: {model.best_params_}')
    print(f'best_score: {model.best_score_:.4f}')


def report_metrics(y_true, y_predict):
    accuracy = metrics.accuracy_score(y_true, y_predict)
    f1 = metrics.f1_score(y_true, y_predict)
    precision = metrics.precision_score(y_true, y_predict)
    recall = metrics.recall_score(y_true, y_predict)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,

    }


def normal_data(x_train, x_test, operation='StandardScaler'):
    if operation == 'MinMaxScaler':
        scaler = preprocessing.MinMaxScaler()

    elif operation == 'Normalizer':
        scaler = preprocessing.Normalizer()
    else:
        scaler = preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test


# pca = PCA(n_components=6)
# x_train = pca.fit_transform(x_train)
# x_test = pca.fit_transform(x_test)

# n_components = min(x_train.shape[1], len(set(y_train)) - 1)
# lda = LinearDiscriminantAnalysis(n_components=n_components)
# x_train = lda.fit_transform(x_train, y_train)
# x_test = lda.transform(x_test)

class ReportModel:
    def __init__(self, path, x_test, y_test, export_name):
        self.path = os.path.join(os.getcwd(), path)
        self.x_test = x_test
        self.y_test = y_test
        self.export_name = export_name

    report_model = {}

    def make_report(self, path_model):
        model = os.path.join(self.path, path_model)
        predict = read_pickle(model, self.x_test, self.y_test)
        name = path_model.split('.')[0]
        print(f'predict:{name}')
        self.report_model[name] = report_metrics(self.y_test, predict)

    def make_chart(self):
        for name, value in self.report_model.items():
            plt.plot(list(value.keys()), list(value.values()), label=name)
        plt.xlabel('model')
        plt.ylabel('metrics')
        plt.title('report_metrics')
        plt.legend()
        return plt

    def __call__(self, *args, **kwargs):
        list_pickle = os.listdir(self.path)
        if list_pickle:
            for model in list_pickle:
                self.make_report(model)
            data = pd.DataFrame(self.report_model)
            output_directory = 'output'
            output_file_name = f'{self.export_name}-{datetime.datetime.now().date()}'
            output_path = os.path.join(os.getcwd(), output_directory, output_file_name)
            data.to_csv(f'{output_path}.csv')
            self.make_chart().savefig(f'{output_path}.png')
        else:
            print('file pickle does not exist "please first train model"')


def read_pickle(model_pickle, x_test, y_test):
    model_ = pickle.load(open(model_pickle, 'rb'))
    predict_ = model_.predict(x_test)
    for index, element in enumerate(predict_):
        print(element, y_test[index])
    print('-' * 6)
    print("  o\n /|\ \n / \ \n^^^^^^^")
    return predict_
