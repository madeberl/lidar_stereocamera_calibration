from tkinter import Tk

from progressbar import progressbar
from sklearn.cluster import KMeans
import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def main(pcd, pcd_object, threshold, name, threshold_ransac):
    object_isolated = compute_distance(pcd, pcd_object, threshold, name)    # remove points appearing in both data
    # showPointCloud(object_isolated, "lidar", False)
    normals_estimated = normal_estimation(object_isolated)                  # estimate normals
    right, left, top = kmeans(normals_estimated)                            # run kmean on object, returns planes detected
    # temp_r = normal_estimation(right)
    # temp_l = normal_estimation(left)
    # temp_t = normal_estimation(top)
    # top, left, right = equation(temp_r, temp_l, temp_t)
    plane_model_a = plane_model_b = plane_model_c = np.empty((0, 4))
    a_in = b_in = c_in = a_out = b_out = c_out = np.empty((0, 3))
    x = []
    for i in progressbar(range(100)):
        a_in, a_out, plane_model_a_x, plane_name_a = ransac(right, threshold_ransac, 3, 500)    # ransac on plane,
        b_in, b_out, plane_model_b_x, plane_name_b = ransac(left, threshold_ransac, 3, 500)     # 3 randomly choosen startpoints
        c_in, c_out, plane_model_c_x, plane_name_c = ransac(top, threshold_ransac, 3, 500)      # 500 iterations
        x.append(i)
        plane_model_a = np.append(plane_model_a, plane_model_a_x, axis=0)
        plane_model_b = np.append(plane_model_b, plane_model_b_x, axis=0)
        plane_model_c = np.append(plane_model_c, plane_model_c_x, axis=0)

    plane_model_a = np.divide(np.sum(plane_model_a, axis=0), len(x))
    plane_model_b = np.divide(np.sum(plane_model_b, axis=0), len(x))
    plane_model_c = np.divide(np.sum(plane_model_c, axis=0), len(x))
    # print(plane_model_a)

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
    schnittpunkt2 = intersection(vektor_rl, 0.4, schnittpunkt)
    schnittpunkt3 = intersection(vektor_rt, -0.35, schnittpunkt)
    schnittpunkt4 = intersection(vektor_lt, -0.45, schnittpunkt)

    schnittpunkt5 = intersection(vektor_lt, -0.45, schnittpunkt3)
    schnittpunkt6 = intersection(vektor_lt, -0.45, schnittpunkt2)
    schnittpunkt7 = intersection(vektor_rt, -0.35, schnittpunkt2)
    s_all = np.concatenate((schnittpunkt, schnittpunkt2, schnittpunkt3, schnittpunkt4, schnittpunkt5, schnittpunkt6,
                            schnittpunkt7))
    # test_function(schnittpunkt2, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 2")
    # test_function(schnittpunkt3, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 3")
    # test_function(schnittpunkt4, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 4")
    # test_function(schnittpunkt5, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 5")
    # test_function(schnittpunkt6, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 6")
    # test_function(schnittpunkt7, plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 7")
    w1 = find_winkel(vektor_rt, vektor_lt, f"R + L {name}")
    w2 = find_winkel(vektor_rl, vektor_rt, f"T + L {name}")
    w3 = find_winkel(vektor_rl, vektor_lt, f"R + T {name}")
    # schnittpunktt = schnittpunkt.reshape(1, 3)
    # schnittpunkt2t = schnittpunkt2.reshape(1, 3)
    # schnittpunkt3t = schnittpunkt3.reshape(1, 3)
    # schnittpunkt4t = schnittpunkt4.reshape(1, 3)
    line_b = np.append(schnittpunkt, schnittpunkt2, axis=0)
    line_r = np.append(schnittpunkt, schnittpunkt3, axis=0)
    line_l = np.append(schnittpunkt, schnittpunkt4, axis=0)

    mesh_1 = getMesh(line_b[:, 0], line_b[:, 1], line_b[:, 2], c='lightgreen', label=w3)
    mesh_2 = getMesh(line_r[:, 0], line_r[:, 1], line_r[:, 2], c='lightsalmon', label=w2)
    mesh_3 = getMesh(line_l[:, 0], line_l[:, 1], line_l[:, 2], c='lightsteelblue', label=w1)

    inlier_1 = getTrace(t_in[:, 0], t_in[:, 1], t_in[:, 2],
                        s=4, c='green', label=f'Top inliers {name}')
    # outlier_1 = getTrace(t_out[:, 0], t_out[:, 1], t_out[:, 2],
    #                     s=4, c='red', label='Top outliers')
    inlier_2 = getTrace(l_in[:, 0], l_in[:, 1], l_in[:, 2],
                        s=4, c='green', label=f'Left inliers {name}')
    # outlier_2 = getTrace(l_out[:, 0], l_out[:, 1], l_out[:, 2],
    #                     s=4, c='red', label='Left outliers')
    inlier_3 = getTrace(r_in[:, 0], r_in[:, 1], r_in[:, 2],
                        s=4, c='green', label=f'Right inliers {name}')
    # outlier_3 = getTrace(r_out[:, 0], r_out[:, 1], r_out[:, 2],
    #                     s=4, c='red', label='Right outliers')

    return s_all, mesh_1, mesh_2, mesh_3, r_in, l_in, t_in


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


def remove_points_lidar(file, cut):
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > cut[0]) & (point[:, 0] < cut[1])
                      & (point[:, 1] > cut[2]) & (point[:, 1] < cut[3])
                      & (point[:, 2] > cut[4]) & (point[:, 2] < cut[5])]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def remove_points_stereo(file):
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > 0.5) & (point[:, 0] < 1.4)
                      & (point[:, 1] > -0.3) & (point[:, 1] < 0.4)
                      & (point[:, 2] > 6.5) & (point[:, 2] < 8)]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def compute_distance(data, data_object, threshold, name):
    # threshold = ([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # i = threshold
    # for i in threshold:
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > threshold)[0]
    object = data_object.select_by_index(ind)
    if name == "Stereo":
        inlier_cloud = statistical_outlier(object)
        return inlier_cloud
    else:
        return object


def radius_outlier(cloud):
    cl, ind = cloud.remove_radius_outlier(nb_points=200, radius=0.5)
    inlier_cloud = cloud.select_by_index(ind)
    display_inlier_outlier(cloud, ind, "radius")
    # bounding_box(inlier_cloud, "radius")
    return inlier_cloud


def statistical_outlier(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=3000, std_ratio=0.01)
    inlier_cloud = cloud.select_by_index(ind)
    display_inlier_outlier(cloud, ind, "statistical")
    # bounding_box(inlier_cloud, "statistical")
    return inlier_cloud


def display_inlier_outlier(cloud, ind, string):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      window_name=string, height=800, width=800)
    return inlier_cloud


def normal_estimation(downpcd):
    downpcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))
    downpcd.orient_normals_consistent_tangent_plane(200)
    return downpcd


def kmeans(pc):
    normals = np.asarray(pc.normals)
    points = np.asarray(pc.points)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10, random_state=100)

    y_kmeans = kmeans.fit_predict(normals)
    # visualising the clusters
    centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
                         s=8, c='yellow', label='Centroids')

    t1 = getTrace(points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2], s=4, c='red',
                  label='Top')  # match with red=1 initial class
    t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
                  label='Left')  # match with green=3 initial class
    t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
                  label='Right')  # match with blue=2 initial class
    # t4 = getTrace(points[y_kmeans == 3, 0], points[y_kmeans == 3, 1], points[y_kmeans == 3, 2], s=4, c='magenta',
    #               label='1')  # match with blue=2 initial class

    # showGraph(
    #     "Oberflächen_kmean",
    #     "Z", "X", "Y",
    #     [t1, t2, t3])  # , centroids])

    top_p = np.stack((points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]), axis=1)
    left_p = np.stack((points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2]), axis=1)
    right_p = np.stack((points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2]), axis=1)

    right_pc = toPointCloud(right_p)
    left_pc = toPointCloud(left_p)
    top_pc = toPointCloud(top_p)
    return right_pc, left_pc, top_pc


def ransac(plane, threshold, n, i):
    plane_model, inliers = plane.segment_plane(distance_threshold=threshold,
                                               ransac_n=n,
                                               num_iterations=i)
    [a, b, c, d] = plane_model
    name = equation(a, b, c)
    # print(f"Plane equation {name}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = plane.select_by_index(inliers)
    outlier_cloud = plane.select_by_index(inliers, invert=True)
    # showPointCloud(inlier_cloud, "inlier", False)
    inlier_cloud = np.asarray(inlier_cloud.points)
    outlier_cloud = np.asarray(outlier_cloud.points)
    # f = open("test_data/tmp.txt", "a+")
    # f.write(f"Plane equation {name}: {a}x + {b}y + {c}z + {d} = 0\n")
    return inlier_cloud, outlier_cloud, np.array([[a, b, c, d]]), name


def getTrace(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def getMesh(x, y, z, c, label, s=4):
    surface_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return surface_points


def equation(plane_x, plane_y, plane_z):
    global plane_name
    if plane_x > 0 and plane_y > 0 and plane_z > 0:
        plane_name = "Right"
    elif plane_x < 0 and plane_z > 0:
        plane_name = "Top"
    elif plane_x < 0 and plane_z < 0:
        plane_name = "Left"
    return plane_name


# def find_equation(plane_model):
#     test = plane_model.points[plane_model.points.is_plane == 1]
#     test = test[['x', 'y', 'z']]
#     plane_top = Plane()
#     plane_top.from_point_cloud(test)
#     equation1 = plane_top.get_equation()
#     [a, b, c, d] = equation1
#     print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
#     return equation1


def finde_intersection(plane1, plane2, plane3, name):
    left_side = [plane1[:3]] + [plane2[:3]] + [plane3[:3]]
    right_side = [[-plane1[3]]] + [[-plane2[3]]] + [[-plane3[3]]]
    i_p = np.linalg.solve(left_side, right_side)
    test_function(i_p, plane1, plane2, plane3, name)
    ip = np.array(i_p.reshape(-1, 3))
    return ip


def test_function(intersection, p1, p2, p3, name):
    test1 = (p1[0] * intersection[0]) + (p1[1] * intersection[1]) + (p1[2] * intersection[2]) + p1[3]
    test2 = (p2[0] * intersection[0]) + (p2[1] * intersection[1]) + (p2[2] * intersection[2]) + p2[3]
    test3 = (p3[0] * intersection[0]) + (p3[1] * intersection[1]) + (p3[2] * intersection[2]) + p3[3]
    print(name)
    if test1 != 0 or test1 != 0.2 or test1 != -0.2:
        print("Testgleichung Plane 1: ", test1)
    if test2 != 0 or test2 != 0.35 or test2 != -0.35:
        print("Testgleichung Plane 2: ", test2)
    if test3 != 0 or test3 != 0.45 or test3 != -0.45:
        print("Testgleichung Plane 3: ", test3)


def intersection(equation, cm, intersect):
    a = np.multiply(equation, cm)
    s = np.subtract(intersect.reshape(-1, 3), a)
    s_shape = np.array(s)
    return s_shape


def showPointCloud(object, name, show_normal):
    if name == '':
        name = "Objekt"
    if show_normal == '':
        show_normal = False
    o3d.visualization.draw_geometries([object], "name", height=800, width=800,
                                      point_show_normal=show_normal)


def normal_distribution(plane_model):
    mean = np.average(plane_model, axis=0)
    print("Mittelwert", mean)


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


def transformate_stereo(ob):
    trans_matrix = np.array([[0., -1., 0.],
                             [0., 0., -1.],
                             [1., 0., 0.]])
    np_object_isolated = np.array(ob.points)
    object1 = np.matmul(np_object_isolated, trans_matrix)
    object1 = toPointCloud(object1)
    return object1


def center_of_mass(l_all, s_all):
    mass_center_l = np.divide(np.sum(l_all, axis=0), len(l_all))
    mass_center_s = np.divide(np.sum(s_all, axis=0), len(l_all))
    q = np.subtract(l_all, mass_center_l)
    p = np.subtract(s_all, mass_center_s)
    q = q.T
    p = p.T
    w = np.dot(q, p.T)
    u_l, s_l, vh_l = np.linalg.svd(w)
    r = np.dot(u_l, vh_l)
    t = np.subtract(mass_center_l, np.dot(r, mass_center_s))
    print("r", r)
    print("t", t)
    return r, t, mass_center_l, mass_center_s


def point_alignments(r, t, mq, mp, rs_in, ls_in, ts_in, s_p):
    rs_in_t = np.add(np.dot(rs_in, r.T), t.T)
    ls_in_t = np.add(np.dot(ls_in, r.T), t.T)
    ts_in_t = np.add(np.dot(ts_in, r.T), t.T)
    s_p_t = np.add(np.dot(s_p, r.T), t.T)
    # rs_in_t = np.add(mq, np.dot(np.subtract(rs_in, mp), r.T))
    # ls_in_t = np.add(mq, np.dot(np.subtract(ls_in, mp), r.T))
    # ts_in_t = np.add(mq, np.dot(np.subtract(ts_in, mp), r.T))
    # s_p_t = np.add(mq, np.dot(np.subtract(s_p, mp), r.T))

    return rs_in_t, ls_in_t, ts_in_t, s_p_t


def inlier_trace(r_in, l_in, t_in, name, color):
    inlier_1 = getTrace(t_in[:, 0], t_in[:, 1], t_in[:, 2],
                        s=4, c=color, label=f'Top inliers {name}')
    # outlier_1 = getTrace(t_out[:, 0], t_out[:, 1], t_out[:, 2],
    #                     s=4, c='red', label='Top outliers')
    inlier_2 = getTrace(l_in[:, 0], l_in[:, 1], l_in[:, 2],
                        s=4, c=color, label=f'Left inliers {name}')
    # outlier_2 = getTrace(l_out[:, 0], l_out[:, 1], l_out[:, 2],
    #                     s=4, c='red', label='Left outliers')
    inlier_3 = getTrace(r_in[:, 0], r_in[:, 1], r_in[:, 2],
                        s=4, c=color, label=f'Right inliers {name}')
    return inlier_1, inlier_2, inlier_3


if __name__ == "__main__":
    """
    Lidar Daten
    """
    file_lidar = "test_data/seq_6.5m_styropor_pos1_0/lidar/1611244442.622.pcd"
    file_object_lidar = "test_data/seq_6.5m_pos1_0/lidar/1611244394.863.pcd"
    """
    Stereo Daten
    """
    file_stero = "test_data/seq_6.5m_styropor_pos1_0/stereo/merged.txt"
    file_object_stereo = "test_data/seq_6.5m_pos1_0/stereo/merged.txt"

    """
    Einlesen der Daten
    """
    file_lidar = o3d.io.read_point_cloud(file_lidar, format='auto')
    file_object_lidar = o3d.io.read_point_cloud(file_object_lidar, format='auto')

    file_stero = o3d.io.read_point_cloud(file_stero, format='xyzrgb')
    file_object_stereo_1 = o3d.io.read_point_cloud(file_object_stereo, format='xyzrgb')

    """ Cropping of data if necessary """
    crop_lidar = [5, 8, -1.5, 2, -0.5, 1]
    crop_stereo = [6.5, 7.5, -0.5, -1.5, 0, -0.5]

    file_lidar = remove_points_lidar(file_lidar, crop_lidar)
    file_object_lidar = remove_points_lidar(file_object_lidar, crop_lidar)

    """
    Stereodaten in Lidar Koordinatensystem überführen 
    """
    file_stereo = transformate_stereo(file_stero)
    file_object_stereo_2 = transformate_stereo(file_object_stereo_1)

    # file_stereo = remove_points_lidar(file_stereo_t, crop_stereo)
    # file_object_stereo = remove_points_lidar(file_object_stereo_2, crop_stereo)

    """ Run main on Lidar and Stereo """
    s_all_l, m1l, m2l, m3l, i1l, i2l, i3l = main(file_lidar, file_object_lidar, 0.1, "Lidar",
                                                 0.03)
    s_all_s, m1s, m2s, m3s, i1s, i2s, i3s = main(file_stereo, file_object_stereo_2, 0.05,
                                                 "Stereo", 0.003)

    """ Compute center of mass and singular value decomposition """
    rotation, translation, mq, mp = center_of_mass(s_all_l, s_all_s)
    """ Allign Stereo data to lidar data """
    i1s, i2s, i3s, s_all_s = point_alignments(rotation, translation, mq, mp, i1s, i2s, i3s, s_all_s)

    """ Build trace for plotly """
    i1l, i2l, i3l = inlier_trace(i1l, i2l, i3l, "Lidar", "green")
    i1s, i2s, i3s = inlier_trace(i1s, i2s, i3s, "Stereo", "orange")
    # i1s = getTrace(i1s[0, :], i1s[1, :], i1s[2, :],
    #                s=4, c='orange', label=f'Right inliers Stereo')
    # i2s = getTrace(i2s[0, :], i2s[1, :], i2s[2, :],
    #                s=4, c='orange', label=f'Left inliers Stereo')
    # i3s = getTrace(i3s[0, :], i3s[1, :], i3s[2, :],
    #                s=4, c='orange', label=f'Top inliers Stereo')
    schnittpunkt1l = getTrace(s_all_l[:, 0], s_all_l[:, 1], s_all_l[:, 2], s=6, c='blue',
                              label=f'S: Lidar')

    schnittpunkt1s = getTrace(s_all_s[:, 0], s_all_s[:, 1], s_all_s[:, 2], s=6, c='red',
                              label=f'S: Stereo')
    showGraph(
        "Oberflächen_ransac open3d",
        "Z", "X", "Y",
        [schnittpunkt1l,
         # schnittpunkt2l, schnittpunkt3l, schnittpunkt4l, schnittpunkt5l, schnittpunkt6l, schnittpunkt7l,
         m1l, m2l, m3l, i1l, i2l, i3l,
         schnittpunkt1s,
         # schnittpunkt2s, schnittpunkt3s, schnittpunkt4s, schnittpunkt5s, schnittpunkt6s, schnittpunkt7s,
         m1s, m2s, m3s, i1s, i2s, i3s])
