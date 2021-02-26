import numpy as np

from scripts.helpers.charts_helperse import accuracy_bar_chart
from scripts.helpers.dictionaries import pose_to_class_num
from scripts.helpers.sklearn_helpers import per_class_accuracy, get_class_percisions_array
from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

per_class_svm = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles.sav',
                                   all_data)
per_class_Tree = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_angles.sav',
                                    all_data)
per_class_Gauss = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_angles.sav',
                                     all_data)

rounded_svm = np.around(per_class_svm, decimals=2)
rounded_tree = np.around(per_class_Tree, decimals=2)
rounded_gauss = np.around(per_class_Gauss, decimals=2)

labels = pose_to_class_num.keys()

accuracy_bar_chart(rounded_svm, labels, "svm")
accuracy_bar_chart(rounded_tree, labels, "tree")
accuracy_bar_chart(rounded_gauss, labels, "gauss")
