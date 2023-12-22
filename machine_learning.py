import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

from main import gird_search, report_metrics, normal_data

data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
data_x = np.array(data.drop(columns='DEATH_EVENT'))
data_y = np.array(data['DEATH_EVENT'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x, data_y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=data_y)

x_train, x_test = normal_data(x_train, x_test)



model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced', random_state=42)
parameters = {
    "n_estimators": [5, 10, 15, 20],
    "max_depth": [1, 2, 3, 5, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}
gird_search(model_random_forest, x_train, x_train, parameters, 'random111')

model_random_forest = pickle.load(open('random111.pickle', 'rb'))
predict_random_forest = model_random_forest.predict(x_test)

for index, element in enumerate(predict_random_forest):
    print(element, y_test[index])

print(report_metrics(y_test, predict_random_forest))
