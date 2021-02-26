import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from scripts.helpers.dictionaries import pose_to_class_num


def train_and_save_model(model, all_data, save_file_name):
    model.fit(all_data.data_train, all_data.labels_train)
    pickle.dump(model, open(save_file_name, 'wb'))


def compare_classifiers(model_array, all_data):
    for classifier in model_array:
        classifier.fit(all_data.data_train, all_data.labels_train)
        print(classifier)
        # print("real value " + str(y_test))
        # print("prediciton " + str(classifier.predict(X_test)))
        scores = accuracy_score(all_data.labels_test, classifier.predict(all_data.data_test))
        print(scores)
        return scores


def load_model_and_predict(model_file, images_data):
    # load the model from disk
    loaded_model = pickle.load(open(model_file, 'rb'))
    # print(images_data)
    result = loaded_model.predict(images_data)
    # print(result)
    return result


def best_hyperparameters(parameters, classifier, data):
    clf = GridSearchCV(classifier, parameters, scoring='accuracy')
    results = clf.fit(data.data_train, data.labels_train)
    print('Best Mean Accuracy: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    optimised_model = clf.best_estimator_
    return optimised_model


def per_class_accuracy(classifier_file, test_data):
    true_labels = test_data.labels_test
    predicted = load_model_and_predict(classifier_file, test_data.data_test)
    disp = confusion_matrix(true_labels, predicted,
                            normalize="true")
    print(disp.diagonal())
    # plt.show()
    return disp.diagonal()


def get_class_percisions_array(accuracy_matrix):
    accuracy = []
    # labels = []
    for k, v in accuracy_matrix.items():
        if pose_to_class_num.__contains__(k):
            accuracy.append(round(v.get('precision'), 2))
            # labels.append(k)
    return accuracy
