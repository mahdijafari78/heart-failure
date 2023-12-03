import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pickle

data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
data_x = np.array(data.drop(columns='DEATH_EVENT'))
data_y = np.array(data['DEATH_EVENT'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x, data_y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=data_y)

# scaler = preprocessing.MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

# scaler = preprocessing.Normalizer()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# pca = PCA(n_components=6)
# x_train = pca.fit_transform(x_train)
# x_test = pca.fit_transform(x_test)


def gird_search(model, parameters, name, scoring='f1'):
    model = model_selection.GridSearchCV(model, parameters, scoring=scoring)
    model.fit(x_train, y_train)
    with open(f'{name}.pickle', 'wb') as f:
        pickle.dump(model, f)


model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=42)
parameters = {
    "n_estimators": [5, 10, 15, 20],
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
gird_search(model_random_forest, parameters, 'random111')

model_random_forest = pickle.load(open('random111.pickle', 'rb'))
predict_random_forest = model_random_forest.predict(x_test)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])


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


print(report_metrics(y_test, predict_random_forest))
