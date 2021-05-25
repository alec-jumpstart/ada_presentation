# these are all very common machine learning libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

from utils import axis_3d

iris_dataset = datasets.load_iris()
x = iris_dataset.data
y = iris_dataset.target

estimators = {
    3: KMeans(n_clusters=3),
    5: KMeans(n_clusters=5),
    8: KMeans(n_clusters=8),
} # number of clusters => estimator

estimator = estimators[8]

figure_1 = plt.figure("Modeled Iris Data", figsize=(8, 6))
# you can ignore pretty much all of these, or play around with them if you're curious
axes = axis_3d(figure=figure_1)

# actually try and fit the data into n clusters
estimator.fit(x)
labels = estimator.labels_

axes.scatter(x[:, 3], x[:, 0], x[:, 2], c=labels.astype(float), edgecolor='k')

# label axes and title
axes.set_xlabel('Petal width (cm)')
axes.set_ylabel('Sepal length (cm)')
axes.set_zlabel('Petal length (cm)')
figure_1.add_axes(axes)

# Plot the correct labels for reference
figure_2 = plt.figure("Correctly Labeled Iris Data", figsize=(8, 6))
axes = axis_3d(figure=figure_2)

actual_labels = [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]
for name, label in actual_labels:
    axes.text3D(
        x[y == label, 3].mean(),
        x[y == label, 0].mean(),
        x[y == label, 2].mean() + 2, name,
        horizontalalignment='center',
        bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
    )

axes.scatter(x[:, 3], x[:, 0], x[:, 2], c=y.astype(float), edgecolor='k')

axes.set_xlabel('Petal width (cm)')
axes.set_ylabel('Sepal length (cm)')
axes.set_zlabel('Petal length (cm)')
figure_2.add_axes(axes)

plt.show()
