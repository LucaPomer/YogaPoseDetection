from sklearn import datasets
from sklearn import svm
from numpy import genfromtxt
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()  # flower dataset
digits = datasets.load_digits()
my_data = genfromtxt('angles.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8,9,10))
classes = genfromtxt('angles.csv', delimiter=',', usecols=(11,))
#my_data = genfromtxt('dataFormatted.csv', delimiter=',')

print(classes)
# classifier = svm.SVC(gamma=0.001, C=100.)
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


trainingData = my_data[:-1]  #all but the last one
targetTrainingData = classes[:-1]#all but the last one

for classifier in classifiers:
    classifier.fit(trainingData, targetTrainingData)
    classifier.predict(my_data[-1:])
    print("prediciton " + str(classifier.predict(my_data[-1:])))




