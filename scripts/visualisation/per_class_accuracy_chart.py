import numpy as np

from scripts.helpers.charts_helperse import accuracy_bar_chart
from scripts.helpers.pose_to_class_dict import pose_to_class_num
from scripts.helpers.sklearn_helpers import per_class_accuracy
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTesting

test_data = MlDataForModelTesting(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/test_data_both.csv')

per_class_svm = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_both.sav',
                                   test_data)
per_class_Tree = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_both.sav',
                                    test_data)
per_class_Gauss = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_both.sav',
                                     test_data)

rounded_svm = np.around(per_class_svm, decimals=2)
rounded_tree = np.around(per_class_Tree, decimals=2)
rounded_gauss = np.around(per_class_Gauss, decimals=2)

labels = pose_to_class_num.keys()

accuracy_bar_chart(rounded_svm, labels, "SVM")
accuracy_bar_chart(rounded_tree, labels, "Decision Tree")
accuracy_bar_chart(rounded_gauss, labels, "Gaussian Process Classifier")
