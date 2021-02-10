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

from scripts.helpers.data_manipulation_helpers import split_data
all_data = pandas.read_csv('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/distance_and_angles.csv')
#print(all_data)
my_data = all_data.values[:, 0:21]
classes = all_data.values[:, 21]
#my_data = genfromtxt('distances.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
#classes = genfromtxt('distances.csv', delimiter=',', usecols=(10,))
# my_data = genfromtxt('dataFormatted.csv', delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(my_data, classes, test_size=0.33, random_state=42)

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

trainingData = my_data[:-1]  # all but the last one
targetTrainingData = classes[:-1]  # all but the last one

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
