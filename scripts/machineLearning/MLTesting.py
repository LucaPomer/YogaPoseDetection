import pickle
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from scripts.helpers.sklearn_helpers import train_and_save_model
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting

train_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/gaus_old')

test_data = MlDataForModelTesting('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/test_gaus_old')
#
# # optimal_svm_angles = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1))
# optimal_svm_angles = SVC(kernel='rbf', C=1)
#
# train_and_save_model(optimal_svm_angles, train_data, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles2.sav')


gauss_both = GaussianProcessClassifier(1**2 * RationalQuadratic(alpha=1, length_scale=1), max_iter_predict=100)
gauss_both.fit(train_data.train_data, train_data.train_labels)
pickle.dump(gauss_both, open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/experiments/gaus_test.sav', 'wb'))

# loaded_model = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/Mlp_optimal_angles.sav', 'rb'))
print(accuracy_score(test_data.test_labels, gauss_both.predict(test_data.test_data)))