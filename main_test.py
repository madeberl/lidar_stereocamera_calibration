from sklearn.cluster import KMeans
import numpy as np
import open3d as o3d
import struct
import time
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from progressbar import progressbar


def main(pcd, pcd_object, threshold, name, threshold_ransac):
    object_isolated = compute_distance(pcd, pcd_object, threshold)
    # showPointCloud(object_isolated, "lidar", False)
    normals_estimated = normal_estimation(object_isolated)
    right, left, top = kmeans(normals_estimated)
    # temp_r = normal_estimation(right)
    # temp_l = normal_estimation(left)
    # temp_t = normal_estimation(top)
    # top, left, right = equation(temp_r, temp_l, temp_t)

    a_in, a_out, plane_model_a, plane_name_a = ransac(right, threshold_ransac, 3, 500)
    b_in, b_out, plane_model_b, plane_name_b = ransac(left, threshold_ransac, 3, 500)
    c_in, c_out, plane_model_c, plane_name_c = ransac(top, threshold_ransac, 3, 500)

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

    w1 = find_winkel(vektor_rt, vektor_lt, f"R + L {name}")
    w2 = find_winkel(vektor_rl, vektor_rt, f"T + L {name}")
    w3 = find_winkel(vektor_rl, vektor_lt, f"R + T {name}")
    w = np.array([[w1, w2, w3]])

    return w


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


def compute_distance(data, data_object, threshold):
    # threshold = ([0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    # i = threshold
    # for i in threshold:
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > threshold)[0]
    object = data_object.select_by_index(ind)
    inlier_cloud = statistical_outlier(object)
    return inlier_cloud


def radius_outlier(cloud):
    cl, ind = cloud.remove_radius_outlier(nb_points=200, radius=0.5)
    inlier_cloud = cloud.select_by_index(ind)
    display_inlier_outlier(cloud, ind, "radius")
    # bounding_box(inlier_cloud, "radius")
    return inlier_cloud


def statistical_outlier(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.1)
    inlier_cloud = cloud.select_by_index(ind)
    # display_inlier_outlier(cloud, ind, "statistical")
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
    # centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2],
    #                      s=8, c='yellow', label='Centroids')

    # t1 = getTrace(points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2], s=4, c='red',
    #               label='Top')  # match with red=1 initial class
    # t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
    #               label='Left')  # match with green=3 initial class
    # t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
    #               label='Right')  # match with blue=2 initial class
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
    return inlier_cloud, outlier_cloud, [a, b, c, d], name


def equation(plane_x, plane_y, plane_z):
    global plane_name
    if plane_x > 0 and plane_y > 0 and plane_z > 0:
        plane_name = "Right"
    elif plane_x < 0 and plane_z > 0:
        plane_name = "Top"
    elif plane_x < 0 and plane_y > 0 and plane_z < 0:
        plane_name = "Left"
    return plane_name


def finde_intersection(plane1, plane2, plane3, name):
    left_side = [plane1[:3]] + [plane2[:3]] + [plane3[:3]]
    right_side = [[-plane1[3]]] + [[-plane2[3]]] + [[-plane3[3]]]
    i_p = np.linalg.solve(left_side, right_side)
    # test_function(i_p, plane1, plane2, plane3, name)
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


def getTrace(x, y, c, label, s=2):
    trace_points = go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def getBar(x, y, label, c):
    bar = go.Bar(
        x=x,
        y=y,
        name=label,
        marker_color=c
    )
    return bar


def showGraph(title, x_colname, y_colname, traces):
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title=x_colname, autorange=True),
            yaxis=dict(title=y_colname, autorange=True)
        )
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(title=title, barmode='group')
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
    # print(f"Winkel {name}:", angle2)
    # angle2 = str(f": {angle2:.2f}")
    return angle2


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
    return r, t


def point_alignments(r, t, rs_in, ls_in, ts_in, s_p):
    rs_in_t = np.dot(r, np.add(rs_in, t.T).T)
    ls_in_t = np.dot(r, np.add(ls_in, t.T).T)
    ts_in_t = np.dot(r, np.add(ts_in, t.T).T)
    s_p_t = np.dot(r, np.add(s_p, t).T)
    return rs_in_t, ls_in_t, ts_in_t, s_p_t


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

    crop_lidar = [5, 8, -1.5, 2, -0.5, 1]
    crop_stereo = [6.5, 7.5, -0.5, -1.5, 0, -0.5]

    file_lidar = o3d.io.read_point_cloud(file_lidar, format='auto')
    file_object_lidar = o3d.io.read_point_cloud(file_object_lidar, format='auto')

    file_stero = o3d.io.read_point_cloud(file_stero, format='xyzrgb')
    file_object_stereo_1 = o3d.io.read_point_cloud(file_object_stereo, format='xyzrgb')

    file_lidar = remove_points_lidar(file_lidar, crop_lidar)
    file_object_lidar = remove_points_lidar(file_object_lidar, crop_lidar)

    file_stereo = transformate_stereo(file_stero)
    file_object_stereo_2 = transformate_stereo(file_object_stereo_1)

    # file_stereo = remove_points_lidar(file_stereo_t, crop_stereo)
    # file_object_stereo = remove_points_lidar(file_object_stereo_2, crop_stereo)
    wl = ws = np.empty((0, 3))
    x = []
    norm_rll = norm_tll = norm_rtl = norm_rls = norm_tls = norm_rts = []
    for i in progressbar(range(100)):
        w_l = main(file_lidar, file_object_lidar, 0.1,
                   "Lidar", 0.005)
        w_s = main(file_stereo, file_object_stereo_2, 0.05,
                   "Stereo", 0.005)
        wl = np.append(wl, w_l, axis=0)
        ws = np.append(ws, w_l, axis=0)
        x.append(i)
        norm_rll.append(np.divide(np.sum(wl[:, 0]), len(x)))
        norm_tll.append(np.divide(np.sum(wl[:, 1]), len(x)))
        norm_rtl.append(np.divide(np.sum(wl[:, 2]), len(x)))
        norm_rls.append(np.divide(np.sum(ws[:, 0]), len(x)))
        norm_tls.append(np.divide(np.sum(ws[:, 1]), len(x)))
        norm_rts.append(np.divide(np.sum(ws[:, 2]), len(x)))

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("Winkel R + L", "Winkel T + L", "Winkel R + T"))

    rll = getBar(x, wl[:, 0], c='indianred', label='Winkel R + L, Lidar')
    tll = getBar(x, wl[:, 1], c='indianred', label='Winkel T + L, Lidar')
    rtl = getBar(x, wl[:, 2], c='indianred', label='Winkel R + T, Lidar')

    rls = getBar(x, ws[:, 0], c='lightsalmon', label='Winkel R + L, Stereo')
    tls = getBar(x, ws[:, 1], c='lightsalmon', label='Winkel T + L, Stereo')
    rts = getBar(x, ws[:, 2], c='lightsalmon', label='Winkel R + T, Stereo')

    norm_rll = getTrace(x, norm_rll, s=4, c='blue', label='Normalverteilung R+L, Lidar')
    norm_tll = getTrace(x, norm_tll, s=4, c='blue', label='Normalverteilung T+L, Lidar')
    norm_rtl = getTrace(x, norm_rtl, s=4, c='blue', label='Normalverteilung R+T, Lidar')
    norm_rls = getTrace(x, norm_rls, s=4, c='lightgreen', label='Normalverteilung R+L, Stereo')
    norm_tls = getTrace(x, norm_tls, s=4, c='lightgreen', label='Normalverteilung T+L, Stereo')
    norm_rts = getTrace(x, norm_rts, s=4, c='lightgreen', label='Normalverteilung R+T, Stereo')

    fig.add_trace(rll, row=1, col=1)
    fig.add_trace(tll, row=2, col=1)
    fig.add_trace(rtl, row=3, col=1)
    fig.add_trace(rls, row=1, col=1)
    fig.add_trace(tls, row=2, col=1)
    fig.add_trace(rts, row=3, col=1)
    fig.add_trace(norm_rll, row=1, col=1)
    fig.add_trace(norm_tll, row=2, col=1)
    fig.add_trace(norm_rtl, row=3, col=1)
    fig.add_trace(norm_rls, row=1, col=1)
    fig.add_trace(norm_tls, row=2, col=1)
    fig.add_trace(norm_rts, row=3, col=1)

    fig.show()

    # showGraph(
    #     "Verteilung Winkel",
    #     "X", "Y",
    #     [rll, tll, rtl, rls, tls, rts, norm_rll, norm_tll, norm_rtl, norm_rls, norm_tls, norm_rts])