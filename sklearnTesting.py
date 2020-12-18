from sklearn import datasets
from sklearn import svm
from numpy import genfromtxt
import numpy as np


iris = datasets.load_iris()  # flower dataset
digits = datasets.load_digits()
my_data = genfromtxt('dataFormatted.csv', delimiter=',', usecols=(0, 24))
classes = genfromtxt('dataFormatted.csv', delimiter=',', usecols=(25, ))
#my_data = genfromtxt('dataFormatted.csv', delimiter=',')
trainingClass =[1, 1, 1, 1, 2, 2, 2, 2, 2]

print(classes)

#print(iris.data[:4])
#classifier = svm.SVC(gamma=0.001, C=100.)

#trainingData = iris.data[:-1]  #all but the last one
#targetTrainingData = iris.target[:-1] #all but the last one

#classifier.fit(trainingData,targetTrainingData)
#classifier.predict(iris.data[-1:])
#print(classifier.predict(iris.data[-1:]))

print(my_data[:4])
classifier = svm.SVC(gamma=0.001, C=100.)

trainingData = my_data[:-1]  #all but the last one
targetTrainingData = classes[:-1]#all but the last one

classifier.fit(trainingData, targetTrainingData)
classifier.predict(my_data[-1:])
print(classifier.predict(my_data[-1:]))