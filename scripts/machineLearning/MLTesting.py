import pickle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from scripts.helpers.sklearn_helpers import train_and_save_model
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining, MlDataForModelTesting

train_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_angles_with_flipped.csv')

test_data = MlDataForModelTesting('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/test_data_angles.csv')

# optimal_svm_angles = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1))
optimal_svm_angles = SVC(kernel='rbf', C=1)

train_and_save_model(optimal_svm_angles, train_data, '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles2.sav')


loaded_model = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/models/SVC_optimal_angles2.sav', 'rb'))
print(accuracy_score(test_data.test_labels, loaded_model.predict(test_data.test_data)))