# these are all very common machine learning libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans

from utils import (add_axis_labels, axis_3d, label_axes_with_actual,
                   scatter_axes)

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

scatter_axes(axes, x, y, labels)

# label axes and title
add_axis_labels(axes)
figure_1.add_axes(axes)

# Plot the correct labels for reference
figure_2 = plt.figure("Correctly Labeled Iris Data", figsize=(8, 6))
axes = axis_3d(figure=figure_2)

label_axes_with_actual(axes, x, y)

scatter_axes(axes, x, y, labels)

add_axis_labels(axes)
figure_2.add_axes(axes)

plt.show()
