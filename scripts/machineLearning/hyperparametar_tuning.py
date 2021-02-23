from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.helpers.sklearn_helpers import best_hyperparameters
from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

parameters_gaus = {'multi_class': ('one_vs_rest', 'one_vs_one')}
parameters_tree = {'max_depth': [5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30]}
parameters_svc = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.025,0.1, 0.5, 1]}
parametersNN = {'activation': ('identity', 'logistic', 'tanh', 'relu'), 'hidden_layer_sizes': [10, 100],
                'max_iter': [1000, 5000]}

parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,), 100],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

gaus = GaussianProcessClassifier(1.0 * Matern(length_scale=1, nu=1.5))
svc = svm.SVC()
mlp = MLPClassifier(max_iter=20000)
tree = DecisionTreeClassifier()

# best_hyperparameters(parameters_gaus, gaus, all_data)
best_hyperparameters(parameter_space, mlp, all_data)