import pandas
from sklearn.model_selection import train_test_split


class MlDataForModelTraining:
    def __init__(self, csv_path):
        csv_data = pandas.read_csv(csv_path, header=None)
        data_as_2d_array = csv_data.values
        num_cols = data_as_2d_array[:1, :].size - 1
        self.train_data = data_as_2d_array[:, 0:num_cols]
        self.train_labels = data_as_2d_array[:, num_cols]


class MlDataForModelTesting:
    def __init__(self, csv_path):
        csv_data = pandas.read_csv(csv_path, header=None)
        data_as_2d_array = csv_data.values
        num_cols = data_as_2d_array[:1, :].size - 1
        self.test_data = data_as_2d_array[:, 0:num_cols]
        self.test_labels = data_as_2d_array[:, num_cols]
