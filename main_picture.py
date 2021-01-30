import os
# from tkinter.filedialog import askopenfilename
from tkinter import Tk
from pyntcloud.geometry.models.plane import Plane
from sklearn.cluster import KMeans
# from sklearn import linear_model
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pyransac3d as pyrsc


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
    point_new = point[(point[:, 0] > 0.5) & (point[:, 0] < 1.4)
                      & (point[:, 1] > -0.3) & (point[:, 1] < 0.4)
                      & (point[:, 2] > 6.5) & (point[:, 2] < 8)]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def compute_distance(data, data_object, name):
    global inlier_cloud
    # threshold = ([0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    i = 0.001
    # for i in np.arange(0.001, 0.01, 0.001):
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > i)[0]
    object = data_object.select_by_index(ind)
    # object = object.voxel_down_sample(voxel_size=0.01)
    inlier_cloud = radius_outlier(object)
    # o3d.io.write_point_cloud("ply_data/cones/distance_radius_" + name + ".ply", inlier_cloud)
    # inlier_cloud = statistical_outlier(object)
    # o3d.visualization.draw_geometries([inlier_cloud], window_name=str(i),
    #                                    height=800, width=800)
    # inlier_cloud = radius_outlier(inlier_cloud)
    return inlier_cloud


def radius_outlier(cloud):
    cl, ind = cloud.remove_radius_outlier(nb_points=60, radius=0.05)
    inlier_cloud = cloud.select_by_index(ind)
    # display_inlier_outlier(cloud, ind, "radius")
    # bounding_box(inlier_cloud, "radius")
    return inlier_cloud


def statistical_outlier(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=1)
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
                  label='Top')  # match with red=1 initial class
    t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
                  label='Right')  # match with green=3 initial class
    t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
                  label='Left')  # match with blue=2 initial class
    # t4 = getTrace(points[y_kmeans == 3, 0], points[y_kmeans == 3, 1], points[y_kmeans == 3, 2], s=4, c='yellow',
    #               label='Rest')  # match with blue=2 initial class
    # showGraph(
    #     "Oberflächen_kmean",
    #     "Z", "X", "Y",
    #     [t1, t2, t3])  # , t4])  # , centroids])

    top_p = np.stack((points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]), axis=1)
    right_p = np.stack((points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2]), axis=1)
    left_p = np.stack((points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2]), axis=1)

    right_pc = toPointCloud(right_p)
    left_pc = toPointCloud(left_p)
    top_pc = toPointCloud(top_p)
    return left_pc, top_pc, right_pc


def ransac(plane, threshold, n, i):
    plane_model, inliers = plane.segment_plane(distance_threshold=threshold,
                                               ransac_n=n,
                                               num_iterations=i)
    [a, b, c, d] = plane_model
    name = equation(a, b)
    print(f"Plane equation {name}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = plane.select_by_index(inliers)
    outlier_cloud = plane.select_by_index(inliers, invert=True)
    inlier_cloud = np.asarray(inlier_cloud.points)
    outlier_cloud = np.asarray(outlier_cloud.points)
    # f = open("test_data/tmp.txt", "a+")
    # f.write(f"Plane equation {name}: {a}x + {b}y + {c}z + {d} = 0\n")
    return inlier_cloud, outlier_cloud, [a, b, c, d], name


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


def getMesh(x, y, z, c, label, s=2):
    surface_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=s, color=c, opacity=1),
        name=label
    )
    return surface_points


def equation(plane_x, plane_y):
    global plane_name
    if plane_x > 0 and plane_y > 0:
        plane_name = "Right"
    elif plane_x < 0 and plane_y < 0:
        plane_name = "Top"
    elif plane_x < 0 and plane_y > 0:
        plane_name = "Left"
    return plane_name


def find_equation(plane_model):
    test = plane_model.points[plane_model.points.is_plane == 1]
    test = test[['x', 'y', 'z']]
    plane_top = Plane()
    plane_top.from_point_cloud(test)
    equation1 = plane_top.get_equation()
    [a, b, c, d] = equation1
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return equation1


def finde_intersection(plane1, plane2, plane3, name):
    left_side = [plane1[:3]] + [plane2[:3]] + [plane3[:3]]
    right_side = [[-plane1[3]]] + [[-plane2[3]]] + [[-plane3[3]]]
    # ip3 = np.linalg.inv(left_side).dot(right_side)
    i_p = np.linalg.solve(left_side, right_side)
    # print(np.allclose(np.dot(left_side, i_p), right_side))
    test_function(i_p, plane1, plane2, plane3, name)
    return i_p


def test_function(intersection, p1, p2, p3, name):
    test1 = (p1[0] * intersection[0]) + (p1[1] * intersection[1]) + (p1[2] * intersection[2]) + p1[3]
    test2 = (p2[0] * intersection[0]) + (p2[1] * intersection[1]) + (p2[2] * intersection[2]) + p2[3]
    test3 = (p3[0] * intersection[0]) + (p3[1] * intersection[1]) + (p3[2] * intersection[2]) + p3[3]
    print(name)
    if test1 != 0:
        print("Testgleichung Plane 1: ", test1)
    if test2 != 0:
        print("Testgleichung Plane 2: ", test2)
    if test3 != 0:
        print("Testgleichung Plane 3: ", test3)


def intersection(equation, cm, intersect):
    global s
    a = np.multiply(np.array(equation[:3]).reshape(-1, 1), cm)
    if equation[0] < 0:
        s = np.subtract(intersect, a)
    else:
        s = np.add(intersect, a)
    # print("Schnittpunkt", s)
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


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T
    p_inter2 = p_inter + aXb_vec
    # print("P_inter1:", p_inter)
    # print("P_inter2:", p_inter2)

    return p_inter[0], (p_inter + aXb_vec)[0], aXb_vec


def find_winkel(plane1, plane2, name):
    plane1 = np.squeeze(np.asarray(plane1))
    plane2 = np.squeeze(np.asarray(plane2))
    nenner = np.dot(plane1[:3], plane2[:3])
    x_modulus = np.sqrt((plane1[:3] * plane1[:3]).sum())
    y_modulus = np.sqrt((plane2[:3] * plane2[:3]).sum())
    cos_angle = nenner / x_modulus / y_modulus
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    print(f"Winkel {name}:", angle2)
    angle2 = str(f": {angle2:.2f}")
    return name + angle2


if __name__ == "__main__":
    Tk().withdraw()
    # file = askopenfilename(initialdir=os.getcwd(), title="File without object")
    # file = "test_data/empty_room_packet.pcd"
    file = "test_data/empty_room_stereo/1611243539.740.txt"
    file_object = "test_data/packet_6.5m_stereo/merged.txt"

    # name2 = file_object.split("/")
    # name2 = name2[-1].split(".")

    pcd = o3d.io.read_point_cloud(file, format='xyzrgb')

    pcd_removed = remove_points(pcd)
    pcd_object = o3d.io.read_point_cloud(file_object, format='xyzrgb')

    pcd_removed_object = remove_points(pcd_object)

    object_isolated = compute_distance(pcd_removed, pcd_removed_object, "test")
    trans_matrix = np.array([[0., -1., 0.],
                             [0., 0., -1.],
                             [1., 0., 0.]])
    np_object_isolated = np.array(object_isolated.points)
    object_isolated = np.matmul(np_object_isolated, trans_matrix)
    object_isolated = toPointCloud(object_isolated)
    normals_estimated = normal_estimation(object_isolated)
    # showPointCloud(normals_estimated, "Normale", True)
    # elbow_method(normals_estimated)
    left, top, right = kmeans(normals_estimated)
    # temp_r = normal_estimation(right)
    # temp_l = normal_estimation(left)
    # temp_t = normal_estimation(top)
    # top, left, right = equation(temp_r, temp_l, temp_t)

    a_in, a_out, plane_model_a, plane_name_a = ransac(right, 0.005, 3, 500)
    b_in, b_out, plane_model_b, plane_name_b = ransac(left, 0.005, 3, 500)
    c_in, c_out, plane_model_c, plane_name_c = ransac(top, 0.005, 3, 500)

    if plane_name_a == "Top":
        t_in, t_out, plane_model_t = a_in, a_out, plane_model_a
    elif plane_name_a == "Left":
        l_in, l_out, plane_model_l = a_in, a_out, plane_model_a
    elif plane_name_a == "Right":
        r_in, r_out, plane_model_r = a_in, a_out, plane_model_a
    if plane_name_b == "Top":
        t_in, t_out, plane_model_t = b_in, b_out, plane_model_b
    elif plane_name_b == "Left":
        l_in, l_out, plane_model_l = b_in, b_out, plane_model_b
    elif plane_name_b == "Right":
        r_in, r_out, plane_model_r = b_in, b_out, plane_model_b
    if plane_name_c == "Top":
        t_in, t_out, plane_model_t = c_in, c_out, plane_model_c
    elif plane_name_c == "Left":
        l_in, l_out, plane_model_l = c_in, c_out, plane_model_c
    elif plane_name_c == "Right":
        r_in, r_out, plane_model_r = c_in, c_out, plane_model_c

    point_rt1, point_rt2, vektor_rt = plane_intersect(plane_model_r, plane_model_t)
    point_lt1, point_lt2, vektor_lt = plane_intersect(plane_model_l, plane_model_t)
    point_rl1, point_rl2, vektor_rl = plane_intersect(plane_model_r, plane_model_l)

    schnittpunkt = finde_intersection(plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 1")
    schnittpunkt2 = intersection(vektor_rl, 0.2, schnittpunkt)
    schnittpunkt3 = intersection(vektor_rt, 0.35, schnittpunkt)
    schnittpunkt4 = intersection(vektor_lt, 0.45, schnittpunkt)

    schnittpunkt5 = intersection(vektor_lt, 0.45, schnittpunkt3)
    schnittpunkt6 = intersection(vektor_lt, 0.45, schnittpunkt2)
    schnittpunkt7 = intersection(vektor_rt, 0.35, schnittpunkt2)
    # test_function(schnittpunkt2, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 2")
    # test_function(schnittpunkt3, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 3")
    # test_function(schnittpunkt4, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 4")
    # test_function(schnittpunkt5, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 5")
    # test_function(schnittpunkt6, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 6")
    # test_function(schnittpunkt7, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 7")
    w1 = find_winkel(vektor_rt, vektor_lt, "R + L")
    w2 = find_winkel(vektor_rl, vektor_rt, "T + L")
    w3 = find_winkel(vektor_rl, vektor_lt, "R + T")
    line_b = []
    line_r = []
    line_l = []
    for i in range(-5, 5, 1):
        line_b.append(np.add(point_rt1, np.multiply(i, np.subtract(point_rt1, point_rt2))))
        line_l.append(np.add(point_rl1, np.multiply(i, np.subtract(point_rl1, point_rl2))))
    for i in range(-5, 5, 1):
        line_r.append(np.add(point_lt1, np.multiply(i, np.subtract(point_lt1, point_lt2))))

    # schnittpunkt_t = intersection(vektor_rl, 0.2, schnittpunkt)
    # schnittpunkt2_t = intersection(vektor_lt, -0.45, schnittpunkt)
    # schnittpunkt3_t = intersection(vektor_rt, -0.35, schnittpunkt)
    # schnittpunkt4_t = intersection(vektor_lt, -0.45, schnittpunkt_t)
    # schnittpunkt5_t = intersection(vektor_rt, -0.35, schnittpunkt_t)
    # schnittpunkt6_t = intersection(vektor_lt, -0.45, schnittpunkt3_t)
    # line_ = np.stack((schnittpunkt, schnittpunkt2_t, schnittpunkt3_t, schnittpunkt6_t), axis=1)
    # line_l = np.stack((schnittpunkt, schnittpunkt2_t, schnittpunkt4_t, schnittpunkt_t), axis=1)
    # line_r = np.stack((schnittpunkt, schnittpunkt_t, schnittpunkt5_t, schnittpunkt3_t), axis=1)
    line_b = np.array(line_b)
    line_r = np.array(line_r)
    line_l = np.array(line_l)

    schnittpunkt1 = getTrace(schnittpunkt[0], schnittpunkt[1], schnittpunkt[2], s=6, c='blue',
                             label='S1: Mitte')
    schnittpunkt2 = getTrace(schnittpunkt2[0], schnittpunkt2[1], schnittpunkt2[2], s=6, c='blue',
                             label='S2: Unten')
    schnittpunkt3 = getTrace(schnittpunkt3[0], schnittpunkt3[1], schnittpunkt3[2], s=6, c='blue',
                             label='S3: Rechts')
    schnittpunkt4 = getTrace(schnittpunkt4[0], schnittpunkt4[1], schnittpunkt4[2], s=6, c='blue',
                             label='S4: Links')
    schnittpunkt5 = getTrace(schnittpunkt5[0], schnittpunkt5[1], schnittpunkt5[2], s=6, c='blue',
                             label='S5: Oben Mitte')
    schnittpunkt6 = getTrace(schnittpunkt6[0], schnittpunkt6[1], schnittpunkt6[2], s=6, c='blue',
                             label='S6: Links Unten')
    schnittpunkt7 = getTrace(schnittpunkt7[0], schnittpunkt7[1], schnittpunkt7[2], s=6, c='blue',
                             label='S7: Rechts unten')

    # s1_t = getTrace(schnittpunkt_t[0], schnittpunkt_t[1], schnittpunkt_t[2], s=8, c='blue',
    #                 label='S1t: Unten')
    # s2_t = getTrace(schnittpunkt2_t[0], schnittpunkt2_t[1], schnittpunkt2_t[2], s=8, c='blue',
    #                 label='S2t: Links')
    # s3_t = getTrace(schnittpunkt3_t[0], schnittpunkt3_t[1], schnittpunkt3_t[2], s=8, c='blue',
    #                 label='S3t: Rechts')
    # s4_t = getTrace(schnittpunkt4_t[0], schnittpunkt4_t[1], schnittpunkt4_t[2], s=8, c='blue',
    #                 label='S4t')
    # s5_t = getTrace(schnittpunkt5_t[0], schnittpunkt5_t[1], schnittpunkt5_t[2], s=8, c='blue',
    #                 label='S5t')
    # s6_t = getTrace(schnittpunkt6_t[0], schnittpunkt6_t[1], schnittpunkt6_t[2], s=8, c='blue',
    #                 label='S6t')

    mesh_1 = getMesh(line_b[:, 0], line_b[:, 1], line_b[:, 2], c='lightgreen', label=w3)
    mesh_2 = getMesh(line_r[:, 0], line_r[:, 1], line_r[:, 2], c='lightsalmon', label=w2)
    mesh_3 = getMesh(line_l[:, 0], line_l[:, 1], line_l[:, 2], c='lightsteelblue', label=w1)

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
        "Oberflächen_ransac open3d",
        "Z", "X", "Y",
        [schnittpunkt1, schnittpunkt2, schnittpunkt3, schnittpunkt4,  # schnittpunkt5,
         mesh_1, mesh_2, mesh_3,
         # s1_t, s2_t, s3_t, s4_t, s5_t, s6_t,
         schnittpunkt5, schnittpunkt6, schnittpunkt7,
         inlier_1, inlier_2, inlier_3])  # outlier_1, outlier_2, outlier_3])
