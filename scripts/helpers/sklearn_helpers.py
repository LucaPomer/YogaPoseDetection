import pickle

from sklearn.model_selection import train_test_split


def train_and_save_model(model, all_data, save_file_name):
    model.fit(all_data.data_train, all_data.labels_train)
    pickle.dump(model, open(save_file_name, 'wb'))


def compare_classifiers(model_array, all_data):
    for classifier in model_array:
        classifier.fit(all_data.data_train, all_data.labels_train)
        print(classifier)
        # print("real value " + str(y_test))
        # print("prediciton " + str(classifier.predict(X_test)))
        print(classifier.score(all_data.data_test, all_data.labels_test))


def load_model_and_predict(model_file, image_data):
    # load the model from disk
    loaded_model = pickle.load(open(model_file, 'rb'))
    print(image_data)
    result = loaded_model.predict(image_data)
    print(result)
    return result
