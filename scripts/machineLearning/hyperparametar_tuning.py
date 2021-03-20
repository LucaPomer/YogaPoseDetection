from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, DotProduct, RationalQuadratic, WhiteKernel
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from scripts.helpers.sklearn_helpers import best_hyperparameters
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining

all_train_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_dist_with_flipped.csv')

parameters_gaus = {'kernel': [1*RBF(), 1*DotProduct(), 1*Matern(length_scale=1, nu=1.5),  1*RationalQuadratic(), 1*WhiteKernel()]}
parameters_tree = {'max_depth': [5, 6, 7, 8 , 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 30]}
parameters_svc = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'), 'C': [0.025,0.1, 0.5, 1]}
parameter_mlp = {
    'hidden_layer_sizes': [(10,30,10),(20,), 100],
    'activation': ['tanh', 'relu', 'identity', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],'max_iter': [5000, 10000]
}

gaus = GaussianProcessClassifier()
svc = svm.SVC()
mlp = MLPClassifier(max_iter=20000)
tree = DecisionTreeClassifier()

optimal_gauss = best_hyperparameters(parameters_gaus, gaus, all_train_data)
optimal_mlp = best_hyperparameters(parameter_mlp, mlp, all_train_data)
optimal_tree = best_hyperparameters(parameters_tree, tree, all_train_data)
optimal_svc = best_hyperparameters(parameters_svc, svc, all_train_data)

