import pickle
from sklearn.metrics import accuracy_score
from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining(
    '/csv_data_files/angles_with_flipped.csv', 0.33, 42)


loaded_model = pickle.load(open('/models/Gauss_optimal_angles.sav', 'rb'))
print(accuracy_score(all_data.labels_test, loaded_model.predict(all_data.data_test)))


