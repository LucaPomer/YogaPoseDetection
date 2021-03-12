import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from scripts.machineLearning.ml_data_for_classification import MlDataForModelTraining

train_data = MlDataForModelTraining(
    '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/train_data_angles_with_flipped.csv')

fig = plt.figure(figsize=(25, 20))
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(train_data.train_data, train_data.train_labels)

print(sklearn.tree.plot_tree(clf))
plt.tight_layout()
plt.show()
