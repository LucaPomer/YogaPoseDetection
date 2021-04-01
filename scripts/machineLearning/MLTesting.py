import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from scripts.helpers.charts_helperse import get_nn_per_class_accuracy
from scripts.helpers.neural_network_helpers import get_model_accuracy
from scripts.helpers.sklearn_helpers import train_and_save_model
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting



test_data_angles = MlDataForModelTesting(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/own_split_test/csv_data_files/test_data_angles.csv')

model_path = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks/correct_split_angles.h5'

get_nn_per_class_accuracy(model_path, test_data_angles)
#
# # optimal_svm_angles = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1))
# optimal_svm_angles = SVC(kernel='rbf', C=1)
#
# train_and_save_model(optimal_svm_angles, train_data, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles2.sav')


# get_model_accuracy(file_path='/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/neural_networks/og_split_angles.h5', mldata_for_testing=test_data)

# gauss_both = GaussianProcessClassifier(1**2 * RationalQuadratic(alpha=1, length_scale=1), max_iter_predict=100)
# gauss_both.fit(train_data.train_data, train_data.train_labels)
# pickle.dump(gauss_both, open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/less_train_data/gaus_test_less_data.sav', 'wb'))
#
# loaded_model = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/less_train_data/gaus_test_less_data.sav', 'rb'))
# print(accuracy_score(test_data.test_labels, loaded_model.predict(test_data.test_data)))