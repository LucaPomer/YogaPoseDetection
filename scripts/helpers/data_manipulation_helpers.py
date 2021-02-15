import csv
from sklearn.model_selection import train_test_split


def write_data(entry_array, file_name):
    with open(file_name, 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(entry_array)


def split_data(data, classes):
    X_train, X_test, y_train, y_test = train_test_split(data, classes, test_size=0.33, random_state=42)

    print(y_train)

