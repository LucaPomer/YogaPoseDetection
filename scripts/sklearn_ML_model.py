import pandas
import sklearn
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scripts.ml_data_for_classification import MlDataForClassification

all_data = MlDataForClassification('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv')


X_train, X_test, y_train, y_test = train_test_split(all_data.data, all_data.class_labels, test_size=0.33, random_state=42)

print(y_train)
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

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    classifier.predict(X_test)
    print(classifier)
    # print("real value " + str(y_test))
    # print("prediciton " + str(classifier.predict(X_test)))
    print(classifier.score(X_test, y_test))




#
#
# print(sklearn.tree.plot_tree(classifiers[4]))
# plt.show()
