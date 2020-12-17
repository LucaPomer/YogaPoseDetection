from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()  # flower dataset
digits = datasets.load_digits()

print(iris.data[:4])
classifier = svm.SVC(gamma=0.001, C=100.)

trainingData = iris.data[:-1]  #all but the last one
targetTrainingData = iris.target[:-1] #all but the last one

classifier.fit(trainingData,targetTrainingData)
classifier.predict(iris.data[-1:])
print(classifier.predict(iris.data[-1:]))