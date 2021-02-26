import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from scripts.ml_data_for_classification import MlDataForModelTraining


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
    all_data_angles = MlDataForModelTraining(
        '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

    all_data_dist = MlDataForModelTraining(
        '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/dist_with_flipped.csv', 0.33, 42)

    all_data_both = MlDataForModelTraining(
        '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/both_with_flipped.csv', 0.33, 42)

    accuracies.append(accuracy_score(all_data_angles.labels_test, angle_model.predict(all_data_angles.data_test)))
    accuracies.append(accuracy_score(all_data_dist.labels_test, dist_model.predict(all_data_dist.data_test)))
    accuracies.append(accuracy_score(all_data_both.labels_test, both_model.predict(all_data_both.data_test)))
    print(accuracies)
    return accuracies
