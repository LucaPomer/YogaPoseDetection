import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

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


def get_per_data_accuacy(angle_model, dist_model, both_model):
    accuracies = []
    test_data_angles = MlDataForModelTesting(
        '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_angles.csv')

    test_data_dist = MlDataForModelTesting(
        '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_dist.csv')

    test_data_both = MlDataForModelTesting(
                     '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_both.csv')

    accuracies.append(accuracy_score(test_data_angles.test_labels, angle_model.predict(test_data_angles.test_data)))
    accuracies.append(accuracy_score(test_data_dist.test_labels, dist_model.predict(test_data_dist.test_data)))
    accuracies.append(accuracy_score(test_data_both.test_labels, both_model.predict(test_data_both.test_data)))
    print(accuracies)
    return accuracies
