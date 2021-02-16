import pandas
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.helpers.angle_calculation import get_keypoint_angles
from scripts.helpers.data_creation_helpers import run_openpose_and_angle_calc
from scripts.helpers.sklearn_helpers import compare_classifiers, train_and_save_model, load_model_and_predict
from scripts.ml_data_for_classification import MlDataForClassification
from scripts.openpose_algorithm import run_openpose_algorithm

all_data = MlDataForClassification('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv', 0.33, 42)

# print(all_data.labels_train)
# classifier = svm.SVC(gamma=0.001, C=100.)
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

# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# clf.score(X_test, y_test)
# print(clf.score(X_test, y_test))

train_and_save_model(classifiers[1], all_data, 'SVC_linear_angles.sav')
# compare_classifiers(classifiers, all_data)

net_res_width = 512
net_res_height = 256

# run_openpose_and_angle_calc('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/unlabled_images', '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_only.csv', net_res_width, net_res_height)
csv_data = pandas.read_csv('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_only.csv', header=None)
data_as_2d_array = csv_data.values
classResult = load_model_and_predict('SVC_linear_angles.sav', data_as_2d_array)
#
#
# print(sklearn.tree.plot_tree(classifiers[4]))
# plt.show()
