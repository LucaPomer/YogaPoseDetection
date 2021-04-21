import pickle
import matplotlib.pyplot as plt
import numpy as np

from scripts.helpers.charts_helperse import get_per_data_accuacy, get_nn_per_data_accuracy, get_nn_per_class_accuracy
from scripts.helpers.pose_to_class_dict import pose_to_class_num
from scripts.helpers.sklearn_helpers import per_class_accuracy
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTesting

test_data = MlDataForModelTesting(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/csv_data_files/test_data_both.csv')

per_class_svm = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/SVC_optimal_both.sav',
                                   test_data)
per_class_Tree = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/Tree_optimal_both.sav',
                                    test_data)
per_class_Gauss = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/Gauss_optimal_both.sav',
                                     test_data)
per_class_mlp = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/correct_split_run_through/models/Mlp_optimal_both.sav', test_data)

per_class_nn = get_nn_per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks/correct_split_both.h5', test_data)

rounded_svm = np.around(per_class_svm, decimals=2)
rounded_tree = np.around(per_class_Tree, decimals=2)
rounded_gauss = np.around(per_class_Gauss, decimals=2)
rounded_mlp = np.around(per_class_mlp, decimals=2)
rounded_nn = np.around(per_class_nn, decimals=2)


labels = pose_to_class_num.keys()


x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()
svm_rects = ax.bar(x - (width *2), rounded_svm, width, label='SVM')
tree_rects = ax.bar(x , rounded_tree, width, label='Tree')
mlp_rects = ax.bar(x + width, rounded_mlp, width, label='MLP')
nn_rects = ax.bar(x + (width*2), rounded_nn, width, label='Neural_Network')
gauss_rects = ax.bar(x - width, rounded_gauss, width, label='Gauss')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Per-Class Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# ax.legend()


plt.tight_layout()

plt.show()