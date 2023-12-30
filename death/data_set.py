import pandas as pd
import numpy as np
from sklearn import model_selection
from main import normal_data
seed = 42
data = pd.read_csv('../dataset/heart_failure_clinical_records_dataset.csv')
data_x = np.array(data.drop(columns='DEATH_EVENT'))
data_y = np.array(data['DEATH_EVENT'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x, data_y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=data_y)

x_train, x_test = normal_data(x_train, x_test)
