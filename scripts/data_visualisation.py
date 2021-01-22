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

my_data = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/anglesNew.csv', delimiter=',',
                     usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
classes = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/anglesNew.csv', delimiter=',', usecols=(11,))

#  OPTION 1: show some of the data for example some specific angles
# plt.scatter(my_data[classes == 1][0], my_data[classes == 1][1], label='class 1', c='red')
# plt.scatter(my_data[classes == 3][0], my_data[classes == 3][1], label='class 3', c='blue')
#
# plt.show()

#  OPTION 2: use Dimensionality Reduction with Principle Component Analysis (PCA)
# pca = sklearn_pca(n_components=2)  # 2-dimensional PCA
# transformed = pd.DataFrame(pca.fit_transform(my_data))
#
# plt.scatter(transformed[classes == 1][0], transformed[classes == 1][1], label='class 1', c='red')
# plt.scatter(transformed[classes == 3][0], transformed[classes == 3][1], label='class 3', c='blue')
# plt.scatter(transformed[classes == 2][0], transformed[classes == 2][1], label='class 2', c='green')
#
# plt.legend()
# plt.show()

#  OPTION 3: use Dimensionality Reduction with Linear Discriminant Analysis
lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))
transformed = pd.DataFrame(lda.fit_transform(my_data, classes))

plt.scatter(transformed[classes == 1][0], transformed[classes == 1][1], label='tree', c='red')
plt.scatter(transformed[classes == 3][0], transformed[classes == 3][1], label='bridge', c='blue')
plt.scatter(transformed[classes == 2][0], transformed[classes == 2][1], label='downwardDog', c='green')

plt.legend(loc=2)
plt.show()