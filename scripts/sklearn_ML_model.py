import numpy as np
import pandas
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scripts.helpers.angle_calculation import get_keypoint_angles
from scripts.helpers.data_creation_helpers import angle_calc_and_write_data, get_angles
from scripts.helpers.sklearn_helpers import compare_classifiers, train_and_save_model, load_model_and_predict, \
    best_hyperparameters, per_class_accuracy
from scripts.ml_data_for_classification import MlDataForModelTraining
from scripts.openpose_algorithm import run_openpose_algorithm

all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * Matern(length_scale=1, nu=1.5)),
    DecisionTreeClassifier(max_depth=9),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

parameters_gaus = {'multi_class': ('one_vs_rest', 'one_vs_one'), 'max_iter_predict':[50,100]}
parameters_tree = {'max_depth': [5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30]}
parameters_svc = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.025,0.1, 0.5, 1]}
parametersNN = {'activation': ('identity', 'logistic', 'tanh', 'relu'), 'hidden_layer_sizes': [10, 100],
                'max_iter': [1000, 5000]}

gaus = GaussianProcessClassifier(1.0 * Matern(length_scale=1, nu=1.5))
svc = svm.SVC()
mlp = MLPClassifier()
tree = DecisionTreeClassifier()

# best_hyperparameters(parameters_gaus, gaus, all_data)


# clf = MLPClassifier(random_state=1, max_iter=400)
#
# gaus.fit(all_data.data_train, all_data.labels_train)
#
# clf.score(all_data.data_test, all_data.labels_test)
# print(gaus.score(all_data.data_test, all_data.labels_test))

## training and saving the optimal models
optimal_svm = SVC(kernel='rbf', C=1)
# train_and_save_model(optimal_svm, all_data, 'SVC_optimal_svm_angles.sav')

per_class_accuracy('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/SVC_optimal_svm_angles.sav', all_data)

## get prediciton from unseen images
# net_res_width = 512
# net_res_height = 256
# images_to_classify = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/unlabled_images'
#
# result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, images_to_classify)
# angles = get_angles(result_from_openpose)
# classResult = load_model_and_predict('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/SVC_optimal_svm_angles.sav', angles)
#



