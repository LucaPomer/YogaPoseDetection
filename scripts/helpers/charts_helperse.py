import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from scripts.helpers.neural_network_helpers import get_model_accuracy
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTesting


def accuracy_bar_chart(accuracy_array, labels, classifier_name):
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects = ax.bar(x, accuracy_array, width, label=classifier_name)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Per Class - ' + classifier_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


def get_per_data_accuacy(angle_model, dist_model, both_model, angles_test_data, dist_test_data, both_test_data):
    accuracies = []
    accuracies.append(accuracy_score(angles_test_data.test_labels, angle_model.predict(angles_test_data.test_data)))
    accuracies.append(accuracy_score(dist_test_data.test_labels, dist_model.predict(dist_test_data.test_data)))
    accuracies.append(accuracy_score(both_test_data.test_labels, both_model.predict(both_test_data.test_data)))
    print(accuracies)
    return accuracies


def get_nn_per_data_accuracy(angle_model_path, dist_model_path, both_model_path, angles_test_data, dist_test_data, both_test_data):
    accuracies = []
    accuracies.append(get_model_accuracy(angle_model_path, angles_test_data))
    accuracies.append(get_model_accuracy(dist_model_path, dist_test_data))
    accuracies.append(get_model_accuracy(both_model_path, both_test_data))
    print(accuracies)
    return accuracies

