from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scripts.helpers.sklearn_helpers import train_and_save_model
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining

train_data_angles = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_angles_with_flipped.csv')

train_data_dist = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_dist_with_flipped.csv')

train_data_both = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_both_with_flipped.csv')


optimal_svm_angles = SVC(kernel='rbf', C=1)
optimal_svm_dist = SVC(kernel='poly', C=1)
optimal_svm_both = SVC(kernel='linear', C=1)
train_and_save_model(optimal_svm_angles, train_data_angles, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles.sav')
train_and_save_model(optimal_svm_dist, train_data_dist, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_dist.sav')
train_and_save_model(optimal_svm_both, train_data_both, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_both.sav')

optimal_tree_angles = DecisionTreeClassifier(max_depth=6)
optimal_tree_dist = DecisionTreeClassifier(max_depth=16)
optimal_tree_both = DecisionTreeClassifier(max_depth=15)
train_and_save_model(optimal_tree_angles, train_data_angles, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_angles.sav')
train_and_save_model(optimal_tree_dist, train_data_dist, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_dist.sav')
train_and_save_model(optimal_tree_both, train_data_both, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Tree_optimal_both.sav')

gauss = GaussianProcessClassifier(1.0 * Matern(length_scale=1, nu=1.5), max_iter_predict=100)
gauss_both = GaussianProcessClassifier(1**2 * RationalQuadratic(alpha=1, length_scale=1), max_iter_predict=100)
train_and_save_model(gauss, train_data_angles, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_angles.sav')
train_and_save_model(gauss, train_data_dist, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_dist.sav')
train_and_save_model(gauss_both, train_data_both, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Gauss_optimal_both.sav')

mlp_angles = MLPClassifier(activation='logistic', alpha=0.0001, hidden_layer_sizes=100, learning_rate='constant',
                    max_iter=5000, solver='adam')
mlp_dist = MLPClassifier(activation='relu', alpha=0.0001, hidden_layer_sizes=100, learning_rate='constant',
                    max_iter=5000, solver='adam')
mlp_both = MLPClassifier(activation='logistic', alpha=0.05, hidden_layer_sizes=100, learning_rate='constant',
                    max_iter=10000, solver='adam')
train_and_save_model(mlp_angles, train_data_angles, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_angles.sav')
train_and_save_model(mlp_dist, train_data_dist, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_dist.sav')
train_and_save_model(mlp_both, train_data_both, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_both.sav')

