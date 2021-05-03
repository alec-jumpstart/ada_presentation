# these are all very common machine learning libraries
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.cluster import KMeans

iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target

estimators = {
    3: KMeans(n_clusters=3),
    5: KMeans(n_clusters=5),
    8: KMeans(n_clusters=8),
} # number of clusters => estimator

estimator = estimators[8]

figure_1 = plt.figure("Modeled Iris Data", figsize=(8, 6))
# you can ignore pretty much all of these, or play around with them if you're curious
axes = Axes3D(figure_1, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)

# actually try and fit the data into n clusters
estimator.fit(X)
labels = estimator.labels_

axes.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float), edgecolor='k')

# label axes and title
axes.set_xlabel('Petal width (cm)')
axes.set_ylabel('Sepal length (cm)')
axes.set_zlabel('Petal length (cm)')
figure_1.add_axes(axes)

# Plot the correct labels for reference
figure_2 = plt.figure("Correctly Labeled Iris Data", figsize=(8, 6))
axes = Axes3D(figure_2, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)

actual_labels = [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]
for name, label in actual_labels:
    axes.text3D(
        X[y == label, 3].mean(),
        X[y == label, 0].mean(),
        X[y == label, 2].mean() + 2, name,
        horizontalalignment='center',
        bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
    )

axes.scatter(X[:, 3], X[:, 0], X[:, 2], c=y.astype(float), edgecolor='k')

axes.set_xlabel('Petal width (cm)')
axes.set_ylabel('Sepal length (cm)')
axes.set_zlabel('Petal length (cm)')
figure_2.add_axes(axes)

plt.show()
