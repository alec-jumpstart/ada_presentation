from mpl_toolkits.mplot3d import Axes3D


def axis_3d(figure):
    return Axes3D(figure, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)

def label_axes_with_actual(axes, x, y):
    actual_labels = [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]
    for name, label in actual_labels:
        axes.text3D(
            x[y == label, 3].mean(),
            x[y == label, 0].mean(),
            x[y == label, 2].mean() + 2, name,
            horizontalalignment='center',
            bbox=dict(alpha=.2, edgecolor='w', facecolor='w')
        )

def scatter_axes(axes, x, y, labels):
    axes.scatter(x[:, 3], x[:, 0], x[:, 2], c=labels.astype(float), edgecolor='k')

def add_axis_labels(axes):
    axes.set_xlabel('Petal width (cm)')
    axes.set_ylabel('Sepal length (cm)')
    axes.set_zlabel('Petal length (cm)')
