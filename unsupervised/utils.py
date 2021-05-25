from mpl_toolkits.mplot3d import Axes3D


def axis_3d(figure):
    return Axes3D(figure, rect=[0, 0, .95, 1], elev=48, azim=134, auto_add_to_figure=False)
