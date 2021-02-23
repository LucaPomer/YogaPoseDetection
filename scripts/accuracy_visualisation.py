from scripts.helpers.charts_helperse import accuracy_bar_chart
from scripts.helpers.dictionaries import pose_to_class_num
from scripts.helpers.sklearn_helpers import per_class_accuracy, get_class_percisions_array
from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

# labels = pose_to_class_num.keys
per_class_svm = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/SVC_optimal_svm_angles.sav',
                                   all_data)
per_class_Tree = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/Tree_optimal_angles.sav',
                                    all_data)
per_class_Gauss = per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/Gauss_optimal_angles.sav',
                                     all_data)

labels = pose_to_class_num.keys()

accuracy_bar_chart(per_class_svm, labels, "svm")
