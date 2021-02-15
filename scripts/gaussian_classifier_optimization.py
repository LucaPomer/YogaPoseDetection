# define model
from numpy import genfromtxt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, Matern, RationalQuadratic, WhiteKernel
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split

my_data = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv', delimiter=',', usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
classes = genfromtxt('/Users/lucapomer/Documents/bachelor/YogaPoseDetection/angles.csv', delimiter=',', usecols=(11,))
# my_data = genfromtxt('dataFormatted.csv', delimiter=',')

X_train, X_test, y_train, y_test = train_test_split(my_data, classes, test_size=0.33, random_state=42)
model = GaussianProcessClassifier()
# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(),  1*RationalQuadratic(), 1*WhiteKernel()]
# define search
search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train, y_train)
# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)
# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))