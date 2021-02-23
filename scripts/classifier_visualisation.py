import pickle

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.helpers.dictionaries import pose_to_class_num
from scripts.ml_data_for_classification import MlDataForModelTraining
from scripts.sklearn_ML_model import tree



all_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)

# fig = plt.figure(figsize=(25,20))
# clf = DecisionTreeClassifier(max_depth=9)
# clf.fit(all_data.data_train, all_data.labels_train)
#
# print(sklearn.tree.plot_tree(clf))
# plt.show()

loaded_model = pickle.load(open('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/SVC_optimal_svm_angles.sav', 'rb'))
# Get support vectors themselves
support_vectors = loaded_model.support_vectors_

# Visualize support vectors

