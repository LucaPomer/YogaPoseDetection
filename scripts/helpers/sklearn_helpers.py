import pickle
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

from scripts.helpers.pose_to_class_dict import pose_to_class_num


def train_and_save_model(model, train_data, save_file_name):
    model.fit(train_data.train_data, train_data.train_labels)
    pickle.dump(model, open(save_file_name, 'wb'))


def compare_classifiers(model_array, train_data, test_data):
    for classifier in model_array:
        classifier.fit(train_data.train_data, train_data.train_labels)
        print(classifier)
        scores = accuracy_score(test_data.test_labels, classifier.predict(test_data.test_data))
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
    results = clf.fit(data.train_data, data.train_labels)
    print('Best Mean Accuracy: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    optimised_model = clf.best_estimator_
    return optimised_model


def per_class_accuracy(classifier_file, test_data):
    true_labels = test_data.test_labels
    predicted = load_model_and_predict(classifier_file, test_data.test_data)
    disp = confusion_matrix(true_labels, predicted,
                            normalize="true")
    print(disp.diagonal())
    return disp.diagonal()


def get_class_percisions_array(accuracy_matrix):
    accuracy = []
    for k, v in accuracy_matrix.items():
        if pose_to_class_num.__contains__(k):
            accuracy.append(round(v.get('precision'), 2))
    return accuracy
