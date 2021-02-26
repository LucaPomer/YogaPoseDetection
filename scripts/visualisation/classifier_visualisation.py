import matplotlib.pyplot as plt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from scripts.ml_data_for_classification import MlDataForModelTraining



all_data = MlDataForModelTraining(
    '/csv_data_files/angles_with_flipped.csv', 0.33, 42)

fig = plt.figure(figsize=(25,20))
clf = DecisionTreeClassifier(max_depth=9)
clf.fit(all_data.data_train, all_data.labels_train)

print(sklearn.tree.plot_tree(clf))
plt.show()



