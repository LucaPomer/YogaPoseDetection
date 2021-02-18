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
from scripts.helpers.data_creation_helpers import angle_calc_and_write_data, get_angles
from scripts.helpers.sklearn_helpers import compare_classifiers, train_and_save_model, load_model_and_predict
from scripts.ml_data_for_classification import MlDataForModelTraining
from scripts.openpose_algorithm import run_openpose_algorithm

all_data = MlDataForModelTraining('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv', 0.33, 42)

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

train_and_save_model(classifiers[1], all_data, 'SVC_sigmoid_angles.sav')
# compare_classifiers(classifiers, all_data)

net_res_width = 512
net_res_height = 256
images_to_classify = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/unlabled_images'

result_from_openpose = run_openpose_algorithm(net_res_width, net_res_height, images_to_classify)
angles = get_angles(result_from_openpose)
classResult = load_model_and_predict('SVC_sigmoid_angles.sav', angles)
print(classifiers[1].score(all_data.data_test, all_data.labels_test))
#
#
# print(sklearn.tree.plot_tree(classifiers[4]))
# plt.show()
