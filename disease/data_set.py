from library import StartDataset

dataset = StartDataset('disease', n=True)
dataset()
x_train, x_test, y_train, y_test = dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test
seed = dataset.seed
