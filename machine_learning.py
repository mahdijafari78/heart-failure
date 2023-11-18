import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
import pickle

data = pd.read_csv('dataset/heart_failure_clinical_records_dataset.csv')
data_x = np.array(data.drop(columns='DEATH_EVENT'))
data_y = np.array(data['DEATH_EVENT'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x, data_y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=data_y)

# best_random_forest = 0
# for i in range(100):
#     model_random_forest = ensemble.RandomForestClassifier(class_weight='balanced',
#                                                           random_state=42)
#     model_random_forest.fit(x_train, y_train)
#     acc_random_forest = model_random_forest.score(x_test, y_test)
#     if acc_random_forest > best_random_forest:
#         best_random_forest = acc_random_forest
#         with open('model_random_forest.pickle', 'wb') as f:
#             pickle.dump(model_random_forest, f)
#
# print('best:', best_random_forest)

model_random_forest = pickle.load(open('model_random_forest.pickle', 'rb'))
predict_random_forest = model_random_forest.predict(x_test)


# for index, element in enumerate(predict_random_forest):
#     print(element, y_test[index])
def report_metrics(name_model, y_predict, y_relly):
    accuracy = metrics.accuracy_score(y_predict, y_relly)
    f1 = metrics.f1_score(y_predict, y_relly)
    precision = metrics.precision_score(y_predict, y_relly)
    recall = metrics.recall_score(y_predict, y_relly)
