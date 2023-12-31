import datetime
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics, preprocessing


def gird_search(model, x_train, y_train, parameters, path, scoring='f1'):
    model = model_selection.GridSearchCV(model, parameters, scoring=scoring)
    path = os.path.join(os.getcwd(), path)
    model.fit(x_train, y_train)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'best_params_: {model.best_params_}')
    print(f'best_score: {model.best_score_:.4f}')


def report_metrics(y_true, y_predict):
    accuracy = metrics.accuracy_score(y_true, y_predict)
    f1 = metrics.f1_score(y_true, y_predict)
    precision = metrics.precision_score(y_true, y_predict)
    recall = metrics.recall_score(y_true, y_predict)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
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
    def __init__(self, path, y_test, export_name):
        self.path = os.path.join(os.getcwd(), path)
        self.y_test = y_test
        self.export_name = export_name

    report_model = {}

    def make_report(self, name):
        predict = pickle.load(open(f'{self.path}/{name}', 'rb'))
        self.report_model[name.split('.')[0]] = report_metrics(self.y_test, predict)

    def make_chart(self):
        for name, value in self.report_model.items():
            plt.plot(list(value.keys()), list(value.values()), label=name)
        plt.xlabel('model')
        plt.ylabel('metrics')
        plt.title('report_metrics')
        plt.legend()
        return plt

    def __call__(self, *args, **kwargs):
        list_predict = os.listdir(self.path)
        if list_predict:
            for predict in list_predict:
                self.make_report(predict)
            data = pd.DataFrame(self.report_model)
            output_directory = 'output'
            output_file_name = f'{self.export_name}-{datetime.datetime.now().date()}'
            output_path = os.path.join(os.getcwd(), output_directory, output_file_name)
            data.to_csv(f'{output_path}.csv')
            self.make_chart().savefig(f'{output_path}.png')
        else:
            raise 'file pickle does not exist'


def make_pickle(m, p, x_test, y_test):
    model_ = pickle.load(open(m, 'rb'))
    predict_ = model_.predict(x_test)
    with open(p, 'wb') as f:
        pickle.dump(predict_, f)

    for index, element in enumerate(predict_):
        print(element, y_test[index])
    print('-' * 6)
    print("  o\n /|\ \n / \ \n^^^^^^^")