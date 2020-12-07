import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from sklearn.cluster import KMeans
import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly


def convert_kitti_bin_to_pcd(bin, name):
    size_float = 4
    list_pcd = []
    with open(bin, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            if 4 <= x <= 30 and -5 <= y <= 5:
                list_pcd.append([x, y, z])
                # list_pcd.append([-y, z, -x])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    o3d.visualization.draw_geometries([pcd], height=800, width=800)
    o3d.io.write_point_cloud("data/" + name + ".ply", pcd)
    return pcd


def remove_points(file):
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > 0) & (point[:, 0] < 22)
                      & (point[:, 1] > -1) & (point[:, 1] < 2)
                      & (point[:, 2] > -0.6) & (point[:, 2] < 1)]
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(point_new)
    return pcd_new


def compute_distance(data, data_object, name):
    threshold = ([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    i = 0.1
    # for i in threshold:
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > i)[0]
    object = data_object.select_by_index(ind)
    # o3d.visualization.draw_geometries([object], window_name=str(i),
    #                                   height=800, width=800)
    # object = object.voxel_down_sample(voxel_size=0.01)
    # inlier_cloud = radius_outlier(object)
    # o3d.io.write_point_cloud("ply_data/cones/distance_radius_" + name + ".ply", inlier_cloud)
    inlier_cloud = statistical_outlier(object)
    # o3d.io.write_point_cloud("ply_data/packet/distance_statistical_" + str(i) + ".ply", inlier_cloud)
    return inlier_cloud


def radius_outlier(cloud):
    cl, ind = cloud.remove_radius_outlier(nb_points=15, radius=0.5)
    inlier_cloud = cloud.select_by_index(ind)
    # display_inlier_outlier(cloud, ind, "radius")
    # bounding_box(inlier_cloud, "radius")
    return inlier_cloud


def statistical_outlier(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.5)
    inlier_cloud = cloud.select_by_index(ind)
    # display_inlier_outlier(cloud, ind, "statistical")
    # bounding_box(inlier_cloud, "statistical")
    return inlier_cloud


def bounding_box(human, string):
    human = human.remove_non_finite_points()
    aligned = human.get_axis_aligned_bounding_box()
    aligned.color = (1, 0, 0)
    # oriented = human.get_oriented_bounding_box()
    # oriented.color = (0, 1, 0)
    # custom_draw_geometry_with_rotation(human, aligned, oriented)
    o3d.visualization.draw_geometries([human, aligned], window_name=string,
                                      height=800, width=800)


def display_inlier_outlier(cloud, ind, string):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      window_name=string, height=800, width=800)
    return inlier_cloud


def custom_draw_geometry_with_rotation(pcd, aligned, oriented):
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(0.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd, aligned, oriented],
                                                              rotate_view, height=800, width=800)


def normal_estimation(downpcd):
    downpcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))
    downpcd.orient_normals_consistent_tangent_plane(200)
    return downpcd


def kmeans(pc):
    normals = np.asarray(pc.normals)
    points = np.asarray(pc.points)

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10, random_state=3)

    y_kmeans = kmeans.fit_predict(normals)

    # visualising the clusters
    centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
                         s=8, c='yellow', label='Centroids')

    t1 = getTrace(points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2], s=4, c='red',
                  label='1')  # match with red=1 initial class
    t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
                  label='2')  # match with green=3 initial class
    t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
                  label='3')  # match with blue=2 initial class

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    showGraph(
        "Oberflächen, kmean",
        "Z", [min(x), max(x)], "X", [min(y), max(y)], "Y", [min(z), max(z) + 0.1],
        [t1, t2, t3, centroids])

    top = np.dstack([points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]])

    return top


def elbow_method(data):
    X = np.asarray(data.points)
    Ks = range(2, 10)
    results = []
    for K in Ks:
        model = KMeans(n_clusters=K)
        model.fit(X)
        results.append(model.inertia_)
    plt.plot(Ks, results, 'o-')
    plt.xlabel("Values of K")
    plt.ylabel("SSE")
    plt.show()


def getTrace(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def showPointCloud(object, name, show_normal):
    if name == '':
        name = "Objekt"
    if show_normal == '':
        show_normal = False
    o3d.visualization.draw_geometries([object], name, height=800, width=800,
                                      point_show_normal=show_normal)


def showGraph(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, traces):
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title=x_colname, range=x_range),
            yaxis=dict(title=y_colname, range=y_range),
            zaxis=dict(title=z_colname, range=z_range)
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, filename='packet_data/test_data/kmean_plot.html')


if __name__ == "__main__":
    Tk().withdraw()
    # file = askopenfilename(initialdir=os.getcwd(), title="File without object")
    file = "packet_data/test_data/empty_room.pcd"
    # file_object = askopenfilename(initialdir=os.getcwd(), title="File with object")
    file_object = "packet_data/test_data/packet_5m.pcd"
    name2 = file_object.split("/")
    name2 = name2[-1].split(".")

    pcd = o3d.io.read_point_cloud(file)
    pcd_removed = remove_points(pcd)
    pcd_object = o3d.io.read_point_cloud(file_object)
    pcd_removed_object = remove_points(pcd_object)

    object_isolated = compute_distance(pcd_removed, pcd_removed_object, "test")
    normals_estimated = normal_estimation(object_isolated)
    # o3d.visualization.draw_geometries([normals_estimated], "Normale", height=800, width=800,
    #                                  point_show_normal=True)
    # elbow_method(normals_estimated)
    area = kmeans(normals_estimated)
    sdf = o3d.geometry.PointCloud()
    sdf.points = o3d.utility.Vector3dVector(area)
    o3d.io.write_point_cloud("tasdf.ply", sdf)
    # o3d.io.write_point_cloud("packet_data/test_data/packet_normalised.ply", normals_estimated)
