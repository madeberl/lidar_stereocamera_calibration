import csv

from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import norm
import open3d as o3d
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from progressbar import progressbar
import plotly.figure_factory as ff


def main(pcd, pcd_object, threshold, name, threshold_ransac):
    object_isolated = compute_distance(pcd, pcd_object, threshold, name)
    normals_estimated = normal_estimation(object_isolated)
    right, left, top = kmeans(normals_estimated)
    plane_model_a = plane_model_b = plane_model_c = np.empty((0, 4))
    vektor_rtc = vektor_ltc = vektor_rlc = np.empty((0, 3))
    a_in = b_in = c_in = a_out = b_out = c_out = np.empty((0, 4))
    for i in progressbar(range(100)):
        a_in, a_out, plane_model_a_x, plane_name_a = ransac(right, threshold_ransac, 3, 500)  # ransac on plane,
        b_in, b_out, plane_model_b_x, plane_name_b = ransac(left, threshold_ransac, 3,
                                                            500)  # 3 randomly choosen startpoints
        c_in, c_out, plane_model_c_x, plane_name_c = ransac(top, threshold_ransac, 3, 500)  # 500 iterations
        plane_model_a = np.append(plane_model_a, plane_model_a_x, axis=0)
        plane_model_b = np.append(plane_model_b, plane_model_b_x, axis=0)
        plane_model_c = np.append(plane_model_c, plane_model_c_x, axis=0)

    # np.savetxt(plane_name_a+name+".txt", plane_model_a)
    # np.savetxt(plane_name_b+name+".txt", plane_model_b)
    # np.savetxt(plane_name_c+name+".txt", plane_model_c)
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

    # for i in progressbar(range(100)):
    #     point_rt1, point_rt2, vektor_rt = plane_intersect(plane_model_r[i, :], plane_model_t[i, :])
    #     point_lt1, point_lt2, vektor_lt = plane_intersect(plane_model_l[i, :], plane_model_t[i, :])
    #     point_rl1, point_rl2, vektor_rl = plane_intersect(plane_model_r[i, :], plane_model_l[i, :])
    #     vektor_rtc = np.append(vektor_rtc, vektor_rt, axis=0)
    #     vektor_ltc = np.append(vektor_ltc, vektor_lt, axis=0)
    #     vektor_rlc = np.append(vektor_rlc, vektor_rl, axis=0)

    # norm_rt, domain_rtc = normal_distribution(plane_model_r, "R+T, " + name)
    # norm_lt, domain_ltc = normal_distribution(plane_model_l, "L+T, " + name)
    # norm_rl, domain_rlc = normal_distribution(plane_model_t, "R+L, " + name)
    w1 = find_winkel(plane_model_l, plane_model_t, "Winkel L + T")
    w2 = find_winkel(plane_model_l, plane_model_r, "Winkel L + R")
    w3 = find_winkel(plane_model_r, plane_model_t, "Winkel R + T")

    return w1, w2, w3


def toPointCloud(points):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


def remove_points(file, cut):
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > cut[0]) & (point[:, 0] < cut[1])
                      & (point[:, 1] > cut[2]) & (point[:, 1] < cut[3])
                      & (point[:, 2] > cut[4]) & (point[:, 2] < cut[5])]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def compute_distance(data, data_object, threshold, name):
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > threshold)[0]
    object = data_object.select_by_index(ind)
    if name == "Stereo":
        inlier_cloud = statistical_outlier(object)
        return inlier_cloud
    else:
        return object


# def radius_outlier(cloud):
#     cl, ind = cloud.remove_radius_outlier(nb_points=200, radius=0.5)
#     inlier_cloud = cloud.select_by_index(ind)
#     display_inlier_outlier(cloud, ind, "radius")
#     # bounding_box(inlier_cloud, "radius")
#     return inlier_cloud


def statistical_outlier(cloud):
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=3000, std_ratio=0.1)
    inlier_cloud = cloud.select_by_index(ind)
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
    inlier_cloud = plane.select_by_index(inliers)
    outlier_cloud = plane.select_by_index(inliers, invert=True)
    inlier_cloud = np.asarray(inlier_cloud.points)
    outlier_cloud = np.asarray(outlier_cloud.points)
    return inlier_cloud, outlier_cloud, np.array([[a, b, c, d]]), name


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
    ip = np.array(i_p.reshape(-1, 3))
    return ip


def intersection(equation, cm, intersect):
    a = np.multiply(equation, cm)
    s = np.subtract(intersect.reshape(-1, 3), a)
    s_shape = np.array(s)
    return s_shape


def getTrace(x, y, c, label, s=2):
    trace_points = go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def getTrace2(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def getBar(x, c):  # , y, label, c):
    bar = go.Histogram(
        x=x,
        marker=dict(color=c),
        xbins=dict(size=0.001),
        hoverinfo="all",
        histfunc="sum",
        histnorm='probability density')
    #     name=label,
    #     marker_color=c
    # )
    return bar


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

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0], np.array([aXb_vec])


def find_winkel(plane1, plane2, name):
    plane1 = np.squeeze(np.asarray(plane1))
    plane2 = np.squeeze(np.asarray(plane2))
    nenner = np.dot(plane1[:3], plane2[:3])
    x_modulus = np.sqrt((plane1[:3] * plane1[:3]).sum())
    y_modulus = np.sqrt((plane2[:3] * plane2[:3]).sum())
    cos_angle = nenner / x_modulus / y_modulus
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    return angle2


def normal_distribution(winkel, name):
    mean = np.mean(winkel, axis=0)
    std = np.std(winkel, axis=0)
    # domain = np.linspace(np.min(winkel, axis=0), np.max(winkel, axis=0))
    # winkel_x = np.sort(winkel[:, 0], axis=0)
    # winkel_y = np.sort(winkel[:, 1], axis=0)
    # winkel_z = np.sort(winkel[:, 2], axis=0)
    # # mean_x = np.mean(winkel_x)
    # # sta_x = np.std(winkel_x)
    # print(f"Mittelwert ist {mean}, Standardabweichung ist {std}")
    # fig, axs = plt.subplots(figsize=(10, 5))
    # for i in range(winkel.shape[1]):
    #     plt.hist(winkel[:, i])
    #     # plt.show()
    #     plt.plot(domain[:, i], norm.pdf(domain[:, i], loc=mean[i], scale=std[i]),
    #              label='$\mathcal{N}$ ' + f'$( \mu \\approx {round(mean[i])} , \sigma \\approx {round(std[i])} )$')
    #     plt.hist(winkel[:, i], edgecolor='black', alpha=.5, density=True)
    #     plt.title(name)
    #     plt.xlabel("Value")
    #     plt.ylabel("Density")
    #     plt.legend()
    #     # axs.set_xlabel('Bins')
    #     # axs.set_ylabel('Count')
    #     # new_winkel = winkel[winkel.x > (mean-std)]
    #     # plt.plot(winkel_y, norm.pdf(winkel_y, mean[1], sta[1]))
    #     # plt.plot(winkel_z, norm.pdf(winkel_z, mean[2], sta[2]))
    #     # print("Percent within one std deviation = ", len(new_winkel)/len(winkel)*100)
    #     plt.show()
    # # print(f"{name}: {domain}")
    # norm_dist_x = norm.pdf(winkel_x, loc=mean[0], scale=std[0])  # * std[0]
    # norm_dist_y = norm.pdf(winkel_y, loc=mean[1], scale=std[1])  # * std[1]
    # norm_dist_z = norm.pdf(winkel_z, loc=mean[2], scale=std[2])  # * std[2]
    # norm_dist = np.stack((norm_dist_x, norm_dist_y, norm_dist_z), axis=1)
    # winkel = np.stack((winkel_x, winkel_y, winkel_z), axis=1)
    return mean, std


def transformate_stereo(ob):
    trans_matrix = np.array([[0., -1., 0.],
                             [0., 0., -1.],
                             [1., 0., 0.]])
    np_object_isolated = np.array(ob.points)
    object1 = np.matmul(np_object_isolated, trans_matrix)
    object1 = toPointCloud(object1)
    return object1


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

    file_lidar = remove_points(file_lidar, crop_lidar)
    file_object_lidar = remove_points(file_object_lidar, crop_lidar)

    file_stereo = transformate_stereo(file_stero)
    file_object_stereo_2 = transformate_stereo(file_object_stereo_1)

    # file_stereo = remove_points_lidar(file_stereo_t, crop_stereo)
    # file_object_stereo = remove_points_lidar(file_object_stereo_2, crop_stereo)
    # norm_rll = norm_tll = norm_rtl = norm_rls = norm_tls = norm_rts = []

    # norm_vrtl, norm_vtll, norm_vrll, v_rtl, v_tll, v_rll, bar_rl, bar_tl, bar_ll
    # norm_vrls, norm_vtls, norm_vrts, v_rts, v_tls, v_rls, bar_rs, bar_ts, bar_ls
    # plane_rl = plane_ll = plane_tl = plane_ls = plane_ts = plane_rs = np.empty((0, 3))
    # for i in progressbar(range(100)):
    plane_rl, plane_ll, plane_tl = main(file_lidar, file_object_lidar, 0.1, "Lidar", 0.03)
    # for i in progressbar(range(100)):
    plane_rs, plane_ls, plane_ts = main(file_stereo, file_object_stereo_2, 0.05, "Stereo", 0.003)

    mean_rl, std_rl = normal_distribution(plane_rl, "R")
    mean_ll, std_ll = normal_distribution(plane_ll, "L")
    mean_tl, std_tl = normal_distribution(plane_tl, "T")
    mean_rs, std_rs = normal_distribution(plane_rs, "R")
    mean_ls, std_ls = normal_distribution(plane_ls, "L")
    mean_ts, std_ts = normal_distribution(plane_ts, "T")

    # fig = ff.create_distplot([plane_rl[:, 0]], bin_size=.0005, group_labels=['R, Lidar, X'], curve_type='normal')

    fig = make_subplots(rows=6, cols=3,
                        subplot_titles=("Flächenvektor a, X", "Lidar, Y", "Z", "Flächenvektor b", "", "",
                                        "Flächenvektor c", "", "", "Flächenvektor a, X", "Stereo, Y", "Z",
                                        "Flächenvektor b", "", "", "Flächenvektor c"))
    plane_rl_x = getBar(plane_rl[:, 0],
                        'blue')  # , norm.pdf(plane_rl[:, 0], mean_rl[0], std_rl[0]), c='blue', label='RxL')
    plane_rl_y = getBar(plane_rl[:, 1],
                        'blue')  # , norm.pdf(plane_rl[:, 1], mean_rl[1], std_rl[1]), c='blue', label='RyL')
    plane_rl_z = getBar(plane_rl[:, 2],
                        'blue')  # , norm.pdf(plane_rl[:, 2], mean_rl[2], std_rl[2]), c='blue', label='RzL')

    plane_ll_x = getBar(plane_ll[:, 0],
                        'blue')  # , norm.pdf(plane_ll[:, 0], mean_ll[0], std_ll[0]), c='blue', label='LxL')
    plane_ll_y = getBar(plane_ll[:, 1],
                        'blue')  # , norm.pdf(plane_ll[:, 1], mean_ll[1], std_ll[1]), c='blue', label='LyL')
    plane_ll_z = getBar(plane_ll[:, 2],
                        'blue')  # , norm.pdf(plane_ll[:, 2], mean_ll[2], std_ll[2]), c='blue', label='LzL')

    plane_tl_x = getBar(plane_tl[:, 0],
                        'blue')  # , norm.pdf(plane_tl[:, 0], mean_tl[0], std_tl[0]), c='blue', label='TxL')
    plane_tl_y = getBar(plane_tl[:, 1],
                        'blue')  # , norm.pdf(plane_tl[:, 1], mean_tl[1], std_tl[1]), c='blue', label='TyL')
    plane_tl_z = getBar(plane_tl[:, 2],
                        'blue')  # , norm.pdf(plane_tl[:, 2], mean_tl[2], std_tl[2]), c='blue', label='TzL')

    plane_rs_x = getBar(plane_rs[:, 0],
                        'blue')  # , norm.pdf(plane_rs[:, 0], mean_rs[0], std_rs[0]), c='blue', label='RxS')
    plane_rs_y = getBar(plane_rs[:, 1],
                        'blue')  # , norm.pdf(plane_rs[:, 1], mean_rs[1], std_rs[1]), c='blue', label='RyS')
    plane_rs_z = getBar(plane_rs[:, 2],
                        'blue')  # , norm.pdf(plane_rs[:, 2], mean_rs[2], std_rs[2]), c='blue', label='RzS')

    plane_ls_x = getBar(plane_ls[:, 0],
                        'blue')  # , norm.pdf(plane_ls[:, 0], mean_ls[0], std_ls[0]), c='blue', label='LxS')
    plane_ls_y = getBar(plane_ls[:, 1],
                        'blue')  # , norm.pdf(plane_ls[:, 1], mean_ls[1], std_ls[1]), c='blue', label='LyS')
    plane_ls_z = getBar(plane_ls[:, 2],
                        'blue')  # , norm.pdf(plane_ls[:, 2], mean_ls[2], std_ls[2]), c='blue', label='LzS')

    plane_ts_x = getBar(plane_ts[:, 0],
                        'blue')  # , norm.pdf(plane_ts[:, 0], mean_ts[0], std_ts[0]), c='blue', label='TxS')
    plane_ts_y = getBar(plane_ts[:, 1],
                        'blue')  # , norm.pdf(plane_ts[:, 1], mean_ts[1], std_ts[1]), c='blue', label='TyS')
    plane_ts_z = getBar(plane_ts[:, 2],
                        'blue')  # , norm.pdf(plane_ts[:, 2], mean_ts[2], std_ts[2]), c='blue', label='TzS')

    fig.add_trace(plane_rl_x, row=1, col=1)
    fig.add_trace(plane_rl_y, row=1, col=2)
    fig.add_trace(plane_rl_z, row=1, col=3)

    fig.add_trace(plane_ll_x, row=2, col=1)
    fig.add_trace(plane_ll_y, row=2, col=2)
    fig.add_trace(plane_ll_z, row=2, col=3)

    fig.add_trace(plane_tl_x, row=3, col=1)
    fig.add_trace(plane_tl_y, row=3, col=2)
    fig.add_trace(plane_tl_z, row=3, col=3)

    fig.add_trace(plane_rs_x, row=4, col=1)
    fig.add_trace(plane_rs_y, row=4, col=2)
    fig.add_trace(plane_rs_z, row=4, col=3)

    fig.add_trace(plane_ls_x, row=5, col=1)
    fig.add_trace(plane_ls_y, row=5, col=2)
    fig.add_trace(plane_ls_z, row=5, col=3)

    fig.add_trace(plane_ts_x, row=6, col=1)
    fig.add_trace(plane_ts_y, row=6, col=2)
    fig.add_trace(plane_ts_z, row=6, col=3)

    norm_rl_x = getTrace(plane_rl[:, 0], norm.pdf(plane_rl[:, 0], mean_rl[0], std_rl[0]), s=4,
                         c='lightcoral', label='RxL')
    norm_rl_y = getTrace(plane_rl[:, 1], norm.pdf(plane_rl[:, 1], mean_rl[1], std_rl[1]), s=4, c='lightcoral',
                         label='RyL')
    norm_rl_z = getTrace(plane_rl[:, 2], norm.pdf(plane_rl[:, 2], mean_rl[2], std_rl[2]), s=4, c='lightcoral',
                         label='RzL')

    norm_ll_x = getTrace(plane_ll[:, 0], norm.pdf(plane_ll[:, 0], mean_ll[0], std_ll[0]), s=4, c='lightcoral',
                         label='RxL')
    norm_ll_y = getTrace(plane_ll[:, 1], norm.pdf(plane_ll[:, 1], mean_ll[1], std_ll[1]), s=4, c='lightcoral',
                         label='RyL')
    norm_ll_z = getTrace(plane_ll[:, 2], norm.pdf(plane_ll[:, 2], mean_ll[2], std_ll[2]), s=4, c='lightcoral',
                         label='RzL')

    norm_tl_x = getTrace(plane_tl[:, 0], norm.pdf(plane_tl[:, 0], mean_tl[0], std_tl[0]), s=4, c='lightcoral',
                         label='RxL')
    norm_tl_y = getTrace(plane_tl[:, 1], norm.pdf(plane_tl[:, 1], mean_tl[1], std_tl[1]), s=4, c='lightcoral',
                         label='RyL')
    norm_tl_z = getTrace(plane_tl[:, 2], norm.pdf(plane_tl[:, 2], mean_tl[2], std_tl[2]), s=4, c='lightcoral',
                         label='RzL')

    norm_rs_x = getTrace(plane_rs[:, 0], norm.pdf(plane_rs[:, 0], mean_rs[0], std_rs[0]), s=4, c='lightcoral',
                         label='RxS')
    norm_rs_y = getTrace(plane_rs[:, 1], norm.pdf(plane_rs[:, 1], mean_rs[1], std_rs[1]), s=4, c='lightcoral',
                         label='RyS')
    norm_rs_z = getTrace(plane_rs[:, 2], norm.pdf(plane_rs[:, 2], mean_rs[2], std_rs[2]), s=4, c='lightcoral',
                         label='RzS')

    norm_ls_x = getTrace(plane_ls[:, 0], norm.pdf(plane_ls[:, 0], mean_ls[0], std_ls[0]), s=4, c='lightcoral',
                         label='RxS')
    norm_ls_y = getTrace(plane_ls[:, 1], norm.pdf(plane_ls[:, 1], mean_ls[1], std_ls[1]), s=4, c='lightcoral',
                         label='RyS')
    norm_ls_z = getTrace(plane_ls[:, 2], norm.pdf(plane_ls[:, 2], mean_ls[2], std_ls[2]), s=4, c='lightcoral',
                         label='RzS')

    norm_ts_x = getTrace(plane_ts[:, 0], norm.pdf(plane_ts[:, 0], mean_ts[0], std_ts[0]), s=4, c='lightcoral',
                         label='RxS')
    norm_ts_y = getTrace(plane_ts[:, 1], norm.pdf(plane_ts[:, 1], mean_ts[1], std_ts[1]), s=4, c='lightcoral',
                         label='RyS')
    norm_ts_z = getTrace(plane_ts[:, 2], norm.pdf(plane_ts[:, 2], mean_ts[2], std_ts[2]), s=4, c='lightcoral',
                         label='RzS')

    fig.add_trace(norm_rl_x, row=1, col=1)
    fig.add_trace(norm_rl_y, row=1, col=2)
    fig.add_trace(norm_rl_z, row=1, col=3)

    fig.add_trace(norm_ll_x, row=2, col=1)
    fig.add_trace(norm_ll_y, row=2, col=2)
    fig.add_trace(norm_ll_z, row=2, col=3)

    fig.add_trace(norm_tl_x, row=3, col=1)
    fig.add_trace(norm_tl_y, row=3, col=2)
    fig.add_trace(norm_tl_z, row=3, col=3)

    fig.add_trace(norm_rs_x, row=4, col=1)
    fig.add_trace(norm_rs_y, row=4, col=2)
    fig.add_trace(norm_rs_z, row=4, col=3)

    fig.add_trace(norm_ls_x, row=5, col=1)
    fig.add_trace(norm_ls_y, row=5, col=2)
    fig.add_trace(norm_ls_z, row=5, col=3)

    fig.add_trace(norm_ts_x, row=6, col=1)
    fig.add_trace(norm_ts_y, row=6, col=2)
    fig.add_trace(norm_ts_z, row=6, col=3)
    fig.update_layout(title={'text': "100 Durchläufe Ransac"})
    fig.show()
