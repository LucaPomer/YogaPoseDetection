from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scripts.helpers.sklearn_helpers import train_and_save_model, compare_classifiers
from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

optimal_svm = SVC(kernel='rbf', C=1)
# train_and_save_model(optimal_svm, all_data, 'SVC_optimal_angles.sav')

optimal_tree = DecisionTreeClassifier(max_depth=9)
# train_and_save_model(optimal_tree, all_data, 'Tree_optimal_angles.sav')

gauss = GaussianProcessClassifier(1.0 * Matern(length_scale=1, nu=1.5))
# train_and_save_model(optimal_tree, all_data, 'Gauss_optimal_angles.sav')



compare_classifiers([optimal_svm, optimal_tree, gauss], all_data)