import math
import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk
from sklearn.cluster import KMeans
from sklearn import linear_model
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
import plotly.graph_objs as go


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
    pcd = toPointCloud(np_pcd)
    o3d.visualization.draw_geometries([pcd], height=800, width=800)
    o3d.io.write_point_cloud("data/" + name + ".ply", pcd)
    return pcd


def toPointCloud(points):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


def remove_points(file):
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > 0) & (point[:, 0] < 22)
                      & (point[:, 1] > -1) & (point[:, 1] < 2)
                      & (point[:, 2] > -0.6) & (point[:, 2] < 1)]
    pcd_new = toPointCloud(point_new)
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
    # o3d.visualization.draw_geometries([human, aligned], window_name=string,
    #                                   height=800, width=800)


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

    z = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    showGraph(
        "Oberflächen_kmean",
        "Z", [min(z), max(z)], "X", [min(x), max(x)], "Y", [min(y), max(y)],
        [t1, t2, t3, centroids])

    right_p = np.stack((points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]), axis=1)
    left_p = np.stack((points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2]), axis=1)
    top_p = np.stack((points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2]), axis=1)

    right_pc = toPointCloud(right_p)
    left_pc = toPointCloud(left_p)
    top_pc = toPointCloud(top_p)
    # print("Right", np.asarray(right_pc.points))
    # print("left", np.asarray(left_pc.points))
    # print("top", np.asarray(top_pc.points))
    return right_pc, left_pc, top_pc


def ransac(plane, threshold, n, i):
    plane_model, inliers = plane.segment_plane(distance_threshold=threshold,
                                               ransac_n=n,
                                               num_iterations=i)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = plane.select_by_index(inliers)
    outlier_cloud = plane.select_by_index(inliers, invert=True)
    inlier_cloud = np.asarray(inlier_cloud.points)
    outlier_cloud = np.asarray(outlier_cloud.points)
    return inlier_cloud, outlier_cloud, [a, b, c, d]


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


def getMesh(xy, z, c, label):
    surface_points = go.Mesh3d(
        x=xy[:, 0], y=xy[:, 1], z=z,
        color=c,
        name=label
    )
    return surface_points


def equation(plane_mod, xy):
    z = (-plane_mod[3] - plane_mod[0] * xy[:, 0] - plane_mod[1] * xy[:, 1]) / plane_mod[2]
    return z


def showPointCloud(object, name, show_normal):
    if name == '':
        name = "Objekt"
    if show_normal == '':
        show_normal = False
    o3d.visualization.draw_geometries([object], "name", height=800, width=800,
                                      point_show_normal=show_normal)


def showGraph(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, traces):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=x_colname, range=x_range),
            yaxis=dict(title=y_colname, range=y_range),
            zaxis=dict(title=z_colname, range=z_range)
        )
    )

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=-2, y=0, z=0.5)
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(scene_camera=camera, title=title)
    # plotly.offline.plot(fig, filename='packet_data/test_data/' + title + '.html')
    fig.show()
    return layout


if __name__ == "__main__":
    Tk().withdraw()
    # file = askopenfilename(initialdir=os.getcwd(), title="File without object")
    file = "test_data/empty_room_packet.pcd"
    # file_object = askopenfilename(initialdir=os.getcwd(), title="File with object")
    file_object = "test_data/packet_5m.pcd"
    name2 = file_object.split("/")
    name2 = name2[-1].split(".")

    pcd = o3d.io.read_point_cloud(file)
    pcd_removed = remove_points(pcd)
    pcd_object = o3d.io.read_point_cloud(file_object)
    # o3d.visualization.draw_geometries([pcd_object], width=800, height=800)
    pcd_removed_object = remove_points(pcd_object)

    object_isolated = compute_distance(pcd_removed, pcd_removed_object, "test")
    normals_estimated = normal_estimation(object_isolated)
    complete_p = pd.DataFrame(normals_estimated.points, columns=["x", "y", "z"])
    # showPointCloud(normals_estimated, "Normale", True)
    # elbow_method(normals_estimated)
    right, left, top = kmeans(normals_estimated)

    r_in, r_out, plane_model_r = ransac(right, 0.002, 3, 500)
    l_in, l_out, plane_model_l = ransac(left, 0.002, 3, 500)
    t_in, t_out, plane_model_t = ransac(top, 0.002, 3, 500)
    # print(plane_model_r[0])

    z_r = equation(plane_model_r, r_in)
    z_l = equation(plane_model_l, l_in)
    z_t = equation(plane_model_t, t_in)

    mesh_1 = getMesh(r_in, z_r, c='lightgreen', label='mesh_1')
    mesh_2 = getMesh(l_in, z_l, c='lightsalmon', label='mesh_2')
    mesh_3 = getMesh(t_in, z_t, c='lightsteelblue', label='mesh_3')

    inlier_1 = getTrace(r_in[:, 0], r_in[:, 1], r_in[:, 2],
                        s=4, c='green', label='inliers_1')
    outlier_1 = getTrace(r_out[:, 0], r_out[:, 1], r_out[:, 2],
                         s=4, c='red', label='outliers_1')
    inlier_2 = getTrace(l_in[:, 0], l_in[:, 1], l_in[:, 2],
                        s=4, c='green', label='inliers_2')
    outlier_2 = getTrace(l_out[:, 0], l_out[:, 1], l_out[:, 2],
                         s=4, c='red', label='outliers_2')
    inlier_3 = getTrace(t_in[:, 0], t_in[:, 1], t_in[:, 2],
                        s=4, c='green', label='inliers_3')
    outlier_3 = getTrace(t_out[:, 0], t_out[:, 1], t_out[:, 2],
                         s=4, c='red', label='outliers_3')
    showGraph(
        "Oberflächen_ransac",
        "Z", [complete_p.x.min(), complete_p.x.max()],
        "X", [complete_p.y.min(), complete_p.y.max()],
        "Y", [complete_p.z.min() - 0.01, complete_p.z.max() + 0.01],
        [inlier_1, outlier_1, inlier_2, outlier_2, inlier_3, outlier_3, mesh_1, mesh_2, mesh_3])