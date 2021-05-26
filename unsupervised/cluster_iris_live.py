import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

from utils import *

iris_dataset = datasets.load_iris()
x = iris_dataset.data
y = iris_dataset.target

estimators = {
    3: KMeans(n_clusters=3),
    5: KMeans(n_clusters=5),
    8: KMeans(n_clusters=8),
}

estimator = estimators[3]

figure_1 = plt.figure("Modeled Iris Data", figsize=(8, 6))

axes = axis_3d(figure=figure_1)

estimator.fit(x)
labels = estimator.labels_

scatter_axes(axes, x, y, labels)

add_axis_labels(axes)
figure_1.add_axes(axes)

figure_2 = plt.figure("Correctly labeled iris data", figsize=(8, 6))
axes = axis_3d(figure=figure_2)

label_axes_with_actual(axes, x, y)

scatter_axes(axes, x, y, labels)

add_axis_labels(axes)
figure_2.add_axes(axes)

plt.show()
