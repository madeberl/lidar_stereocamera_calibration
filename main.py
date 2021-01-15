# import os
# from tkinter.filedialog import askopenfilename
from tkinter import Tk
from sklearn.cluster import KMeans
# from sklearn import linear_model
# from pyntcloud import PyntCloud
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
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10, random_state=10)

    y_kmeans = kmeans.fit_predict(normals)
    # visualising the clusters
    # centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
    #                     s=8, c='yellow', label='Centroids')

    t1 = getTrace(points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2], s=4, c='red',
                  label='Top')  # match with red=1 initial class
    t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
                  label='Left')  # match with green=3 initial class
    t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
                  label='Right')  # match with blue=2 initial class

    z = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    showGraph(
        "Oberflächen_kmean",
        "Z", "X", "Y",
        [t1, t2, t3])  # , centroids])

    top_p = np.stack((points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]), axis=1)
    left_p = np.stack((points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2]), axis=1)
    right_p = np.stack((points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2]), axis=1)

    right_pc = toPointCloud(right_p)
    left_pc = toPointCloud(left_p)
    top_pc = toPointCloud(top_p)
    return right_pc, left_pc, top_pc


def ransac(plane, threshold, n, i, name):
    plane_model, inliers = plane.segment_plane(distance_threshold=threshold,
                                               ransac_n=n,
                                               num_iterations=i)
    [a, b, c, d] = plane_model
    print(f"Plane equation {name}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
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


def getMesh(x, y, z, c, label):
    surface_points = go.Scatter3d(
        x=x, y=y, z=z,
        line=dict(color=c, width=8),
        name=label,
        mode='lines'
    )
    return surface_points


def equation(plane1, plane2, plane3, in1, in2, in3, out1, out2, out3):
    global a, b, c, a1, a2, b2, b1, c1, c2
    plane = [plane1] + [plane2] + [plane3]
    in_a = [in1] + [in2] + [in3]
    out_a = [out1] + [out2] + [out3]
    for i, val in enumerate(plane):
        if plane[i][0] < 0 and plane[i][1] < 0:
            a = plane[i]
            a1 = in_a[i]
            a2 = out_a[i]
        elif plane[i][0] > 0 and plane[i][1] < 0:
            b = plane[i]
            b1 = in_a[i]
            b2 = out_a[i]
        elif plane[i][0] > 0 and plane[i][1] > 0:
            c = plane[i]
            c1 = in_a[i]
            c2 = out_a[i]
    return a, b, c, a1, b1, c1, a2, b2, c2


def test_function(plane1, plane2, plane3):
    left_side = [plane1[:3]] + [plane2[:3]] + [plane3[:3]]
    right_side = [[-plane1[3]]] + [[-plane2[3]]] + [[-plane3[3]]]
    # np.linalg.inv(left_side).dot(right_side)
    schnittpunkt = np.linalg.solve(left_side, right_side)
    # print(np.allclose(np.dot(left_side, schnittpunkt), right_side))
    print("Schnittpunkt:", schnittpunkt)
    print("Testgleichung Plane 1: ",
          (plane1[0] * schnittpunkt[0]) + (plane1[1] * schnittpunkt[1]) + (plane1[2] * schnittpunkt[2]) + plane1[3])
    print("Testgleichung Plane 2: ",
          (plane2[0] * schnittpunkt[0]) + (plane2[1] * schnittpunkt[1]) + (plane2[2] * schnittpunkt[2]) + plane2[3])
    print("Testgleichung Plane 3: ",
          (plane3[0] * schnittpunkt[0]) + (plane3[1] * schnittpunkt[1]) + (plane3[2] * schnittpunkt[2]) + plane3[3])
    return schnittpunkt


def intersection(equation, cm, intersect):
    global s
    a = np.multiply(np.array(equation[:3]).reshape(-1, 1), cm)
    if equation[0] < 0:
        s = np.subtract(intersect, a)
    else:
        s = np.add(intersect, a)
    print("Schnittpunkt", s)
    return s


def showPointCloud(object, name, show_normal):
    if name == '':
        name = "Objekt"
    if show_normal == '':
        show_normal = False
    o3d.visualization.draw_geometries([object], "name", height=800, width=800,
                                      point_show_normal=show_normal)


def showGraph(title, x_colname, y_colname, z_colname, traces):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=x_colname, autorange=True),
            yaxis=dict(title=y_colname, autorange=True),
            zaxis=dict(title=z_colname, autorange=True)
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

    a_in, a_out, plane_model_a = ransac(right, 0.001, 3, 500, "a")
    b_in, b_out, plane_model_b = ransac(left, 0.001, 3, 500, "b")
    c_in, c_out, plane_model_c = ransac(top, 0.001, 3, 500, "c")
    plane_model_t, plane_model_l, plane_model_r, t_in, l_in, r_in, t_out, l_out, r_out = equation(
        plane_model_a, plane_model_b, plane_model_c,
        a_in, b_in, c_in,
        a_out, b_out, c_out)

    schnittpunkt = test_function(plane_model_r, plane_model_l, plane_model_t)

    schnittpunkt2 = intersection(plane_model_t, 0.2, schnittpunkt)
    schnittpunkt3 = intersection(plane_model_l, 0.45, schnittpunkt)
    schnittpunkt4 = intersection(plane_model_r, 0.35, schnittpunkt)

    schnittpunkt1t = getTrace(schnittpunkt[0], schnittpunkt[1], schnittpunkt[2], s=6, c='blue',
                              label='Schnittpunkt Mitte')
    schnittpunkt2t = getTrace(schnittpunkt2[0], schnittpunkt2[1], schnittpunkt2[2], s=6, c='blue',
                              label='Schnittpunkt Unten')
    schnittpunkt3t = getTrace(schnittpunkt3[0], schnittpunkt3[1], schnittpunkt3[2], s=6, c='blue',
                              label='Schnittpunkt Rechts')
    schnittpunkt4t = getTrace(schnittpunkt4[0], schnittpunkt4[1], schnittpunkt4[2], s=6, c='blue',
                              label='Schnittpunkt Links')

    line_b = np.concatenate((schnittpunkt, schnittpunkt2), axis=1)
    line_r = np.concatenate((schnittpunkt, schnittpunkt3), axis=1)
    line_l = np.concatenate((schnittpunkt, schnittpunkt4), axis=1)

    mesh_1 = getMesh(line_b[0], line_b[1], line_b[2], c='lightgreen', label='mesh_1')
    mesh_2 = getMesh(line_r[0], line_r[1], line_r[2], c='lightsalmon', label='mesh_2')
    mesh_3 = getMesh(line_l[0], line_l[1], line_l[2], c='lightsteelblue', label='mesh_3')

    inlier_1 = getTrace(t_in[:, 0], t_in[:, 1], t_in[:, 2],
                        s=4, c='green', label='Top inliers')
    # outlier_1 = getTrace(t_out[:, 0], t_out[:, 1], t_out[:, 2],
    #                     s=4, c='red', label='Top outliers')
    inlier_2 = getTrace(l_in[:, 0], l_in[:, 1], l_in[:, 2],
                        s=4, c='green', label='Left inliers')
    # outlier_2 = getTrace(l_out[:, 0], l_out[:, 1], l_out[:, 2],
    #                     s=4, c='red', label='Left outliers')
    inlier_3 = getTrace(r_in[:, 0], r_in[:, 1], r_in[:, 2],
                        s=4, c='green', label='Right inliers')
    # outlier_3 = getTrace(r_out[:, 0], r_out[:, 1], r_out[:, 2],
    #                     s=4, c='red', label='Right outliers')
    showGraph(
        "Oberflächen_ransac",
        "Z", "X", "Y",
        [schnittpunkt1t, schnittpunkt2t, schnittpunkt3t, schnittpunkt4t, inlier_1, inlier_2, inlier_3,
         mesh_1, mesh_2, mesh_3])  # outlier_1, outlier_2, outlier_3])
