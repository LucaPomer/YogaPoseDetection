import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from scripts.helpers.dictionaries import pose_to_class_num
from scripts.openpose_algorithm import run_openpose_algorithm


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


def load_model_and_predict(model_file, images_data):
    # load the model from disk
    loaded_model = pickle.load(open(model_file, 'rb'))
    print(images_data)
    result = loaded_model.predict(images_data)
    print(result)
    return result


def best_hyperparameters(parameters, classifier, data):
    clf = GridSearchCV(classifier, parameters)
    results = clf.fit(data.data, data.class_labels)
    print('Best Mean Accuracy: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)


def per_class_accuracy(classifier_file, test_data):
    true_labels = test_data.labels_test
    predicted = load_model_and_predict(classifier_file, test_data.data_test)
    report = classification_report(true_labels, predicted, target_names=pose_to_class_num.keys())
    print(report)
    return report
