import pandas
from sklearn.model_selection import train_test_split


class MlDataForClassification:
    def __init__(self, csv_path, test_size, random_state):
        csv_data = pandas.read_csv(csv_path, header=None)
        data_as_2d_array = csv_data.values
        num_cols = data_as_2d_array[:1, :].size - 1
        self.data = data_as_2d_array[:, 0:num_cols]
        self.class_labels = data_as_2d_array[:, num_cols]
        self.data_train, self.data_test, self.labels_train, self.labels_test = train_test_split(self.data,
                                                                                                self.class_labels,
                                                                                                test_size=test_size,
                                                                                                random_state=random_state)
