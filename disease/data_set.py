import pandas as pd
import numpy as np
from sklearn import model_selection
from library import normal_data
import os

seed = 42
path = os.path.join(os.getcwd(), 'dataset/heart.csv')
data = pd.read_csv(path)
data_x = np.array(data.drop(columns='target'))
data_y = np.array(data['target'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(data_x, data_y, test_size=0.2, random_state=42,
                                                                    shuffle=True, stratify=data_y)

x_train, x_test = normal_data(x_train, x_test)
