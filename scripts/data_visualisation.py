import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearn_pca
from numpy import genfromtxt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# csv_file = '/Users/lucapomer/Documents/bachelor/YogaPoseDetection/accuracy.csv'
# data = pd.read_csv(csv_file)
# net_width = data["height"]
# total_accuracy = data["total_accuracy"]
#
# x = list(net_width)
# y = list(total_accuracy)
#
# plt.bar(x, y)
# plt.xlabel('height')
# plt.ylabel('Total_accuracy')
# plt.title('Data')
# plt.show()
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.ml_data_for_classification import MlDataForModelTraining

all_data = MlDataForModelTraining('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/csv_data_files/angles_with_flipped.csv', 0.33, 42)


#  OPTION 1: show some of the data for example some specific angles
# plt.scatter(all_data.data[all_data.class_labels == 1][0], all_data.data[all_data.class_labels == 1][1], label='tree', c='red')
# plt.scatter(all_data.data[all_data.class_labels == 3][0], all_data.data[all_data.class_labels == 3][1], label='bridge', c='blue')
#
# plt.show()

#  OPTION 2: use Dimensionality Reduction with Principle Component Analysis (PCA)
# pca = sklearn_pca(n_components=2)  # 2-dimensional PCA
# transformed = pd.DataFrame(pca.fit_transform(all_data.data, all_data.class_labels))
# #
# plt.scatter(transformed[all_data.class_labels == 1][0], transformed[all_data.class_labels == 1][1], label='tree' )
# plt.scatter(transformed[all_data.class_labels == 2][0], transformed[all_data.class_labels == 2][1], label='downwarddog' )
# plt.scatter(transformed[all_data.class_labels == 3][0], transformed[all_data.class_labels == 3][1], label='bridge')
# plt.scatter(transformed[all_data.class_labels == 4][0], transformed[all_data.class_labels == 4][1], label='child')
# plt.scatter(transformed[all_data.class_labels == 5][0], transformed[all_data.class_labels == 5][1], label='mountain')
# plt.scatter(transformed[all_data.class_labels == 6][0], transformed[all_data.class_labels == 6][1], label='plank')
# plt.scatter(transformed[all_data.class_labels == 7][0], transformed[all_data.class_labels == 7][1], label='seatedforward')
# plt.scatter(transformed[all_data.class_labels == 8][0], transformed[all_data.class_labels == 8][1], label='triangle')
# plt.scatter(transformed[all_data.class_labels == 9][0], transformed[all_data.class_labels == 9][1], label='warrior1')
# plt.scatter(transformed[all_data.class_labels == 10][0], transformed[all_data.class_labels == 10][1], label='warrior2')
# #
# plt.legend()
# plt.show()

#  OPTION 3: use Dimensionality Reduction with Linear Discriminant Analysis
# lda = make_pipeline(StandardScaler(),
#                     LinearDiscriminantAnalysis(n_components=2))
# transformed = pd.DataFrame(lda.fit_transform(all_data.data, all_data.class_labels))
#
# plt.scatter(transformed[all_data.class_labels == 1][0], transformed[all_data.class_labels == 1][1], label='tree' )
# plt.scatter(transformed[all_data.class_labels == 2][0], transformed[all_data.class_labels == 2][1], label='downwarddog' )
# plt.scatter(transformed[all_data.class_labels == 3][0], transformed[all_data.class_labels == 3][1], label='bridge')
# plt.scatter(transformed[all_data.class_labels == 4][0], transformed[all_data.class_labels == 4][1], label='child')
# plt.scatter(transformed[all_data.class_labels == 5][0], transformed[all_data.class_labels == 5][1], label='mountain')
# plt.scatter(transformed[all_data.class_labels == 6][0], transformed[all_data.class_labels == 6][1], label='plank')
# plt.scatter(transformed[all_data.class_labels == 7][0], transformed[all_data.class_labels == 7][1], label='seatedforward')
# plt.scatter(transformed[all_data.class_labels == 8][0], transformed[all_data.class_labels == 8][1], label='triangle')
# plt.scatter(transformed[all_data.class_labels == 9][0], transformed[all_data.class_labels == 9][1], label='warrior1')
# plt.scatter(transformed[all_data.class_labels == 10][0], transformed[all_data.class_labels == 10][1], label='warrior2')
#
# plt.legend(loc=2)
# plt.show()