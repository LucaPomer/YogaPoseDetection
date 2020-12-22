from sklearn import datasets
from sklearn import svm
from numpy import genfromtxt
import numpy as np


iris = datasets.load_iris()  # flower dataset
digits = datasets.load_digits()
my_data = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/data_with_angles.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
classes = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/data_with_angles.csv', delimiter=',', usecols=(9,))
#my_data = genfromtxt('dataFormatted.csv', delimiter=',')

print(classes)


print(my_data[0])
classifier = svm.SVC(gamma=0.001, C=100.)

trainingData = my_data[:-1]  #all but the last one
targetTrainingData = classes[:-1]#all but the last one

classifier.fit(trainingData, targetTrainingData)
classifier.predict(my_data[-1:])
print(classifier.predict(my_data[-1:]))