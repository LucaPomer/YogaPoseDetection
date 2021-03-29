import keras
from keras import models


def get_model_accuracy(file_path, mldata_for_testing):
    model = models.load_model(file_path)
    num_classes = 10
    y_test = keras.utils.to_categorical(mldata_for_testing.test_labels, num_classes)
    loss, accuracy = model.evaluate(mldata_for_testing.test_data, y_test, verbose=False)
    return accuracy
