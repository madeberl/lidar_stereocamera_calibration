import argparse
import struct
import time

import numpy as np
import open3d as o3d
import plotly.graph_objs as go
from scipy.spatial import distance
from sklearn.cluster import KMeans

global debug


def main(pcd, pcd_object, threshold, name, threshold_ransac):
    object_isolated = compute_distance(pcd, pcd_object, threshold, name)  # remove points appearing in both data

    normals_estimated = normal_estimation(object_isolated)  # estimate normals
    right, left, top = kmeans(normals_estimated, name)  # run kmean on object, returns planes detected

    plane_model_a = plane_model_b = plane_model_c = plane_model_r = plane_model_t = plane_model_l = np.empty((0, 4))
    a_in = b_in = c_in = a_out = b_out = c_out = r_in = l_in = t_in = np.empty((0, 3))
    x = []
    plane_name_a = plane_name_b = plane_name_c = ''
    """
    Ransac on plane, with 3 randomly chosen start points and 500 iterations
    """
    for i in range(100):
        a_in, a_out, plane_model_a_, plane_name_a = ransac(right, threshold_ransac, 3, 1000)
        b_in, b_out, plane_model_b_, plane_name_b = ransac(left, threshold_ransac, 3, 1000)
        c_in, c_out, plane_model_c_, plane_name_c = ransac(top, threshold_ransac, 3, 1000)
        x.append(i)
        plane_model_a = np.append(plane_model_a, plane_model_a_, axis=0)
        plane_model_b = np.append(plane_model_b, plane_model_b_, axis=0)
        plane_model_c = np.append(plane_model_c, plane_model_c_, axis=0)

    """
    Take mean of 100 ransac iterations
    """
    plane_model_a = np.divide(np.sum(plane_model_a, axis=0), len(x))
    plane_model_b = np.divide(np.sum(plane_model_b, axis=0), len(x))
    plane_model_c = np.divide(np.sum(plane_model_c, axis=0), len(x))

    if debug:
        inl1 = getTrace(a_in[:, 0], a_in[:, 1], a_in[:, 2], c="green", s=4, label=f"{plane_name_a} inliers")
        inl2 = getTrace(b_in[:, 0], b_in[:, 1], b_in[:, 2], c="green", s=4, label=f"{plane_name_b} inliers")
        inl3 = getTrace(c_in[:, 0], c_in[:, 1], c_in[:, 2], c="green", s=4, label=f"{plane_name_c} inliers")
        out1 = getTrace(a_out[:, 0], a_out[:, 1], a_out[:, 2], c="red", s=4, label=f"{plane_name_a} outliers")
        out2 = getTrace(b_out[:, 0], b_out[:, 1], b_out[:, 2], c="red", s=4, label=f"{plane_name_b} outliers")
        out3 = getTrace(c_out[:, 0], c_out[:, 1], c_out[:, 2], c="red", s=4, label=f"{plane_name_c} outliers")
        showGraph(f"RANSAC {name}",
                  "Z", "X", "Y",
                  [inl1, inl2, inl3, out1, out2, out3])

    """
    Planes will be assigned to side
    """
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

    inliers = np.concatenate((t_in, l_in, r_in))

    """
    Find intersections line
    """
    vektor_rt = plane_intersect(plane_model_r, plane_model_t)
    vektor_lt = plane_intersect(plane_model_l, plane_model_t)
    vektor_rl = plane_intersect(plane_model_r, plane_model_l)

    """
    Find intersections
    """
    """ Intersection of all 3 planes """
    schnittpunkt = finde_intersection(plane_model_r, plane_model_l, plane_model_t, "Schnittpunkt 1")
    """ Intersection down """
    schnittpunkt2 = intersection(vektor_rl, 0.4, schnittpunkt)
    """ Intersection right top """
    schnittpunkt3 = intersection(vektor_rt, -0.35, schnittpunkt)
    """ Intersection left top """
    schnittpunkt4 = intersection(vektor_lt, -0.45, schnittpunkt)
    """ Intersection back top """
    schnittpunkt5 = intersection(vektor_lt, -0.45, schnittpunkt3)
    """ Intersection right down """
    schnittpunkt6 = intersection(vektor_rt, -0.35, schnittpunkt2)
    """ Intersection left down """
    schnittpunkt7 = intersection(vektor_lt, -0.45, schnittpunkt2)
    s_all = np.concatenate((schnittpunkt, schnittpunkt2, schnittpunkt3, schnittpunkt4, schnittpunkt5, schnittpunkt6,
                            schnittpunkt7))
    if debug:
        """
        return angle in degrees
        """
        find_winkel(vektor_rt, vektor_lt, f"R + L {name}")
        find_winkel(vektor_rl, vektor_rt, f"T + L {name}")
        find_winkel(vektor_rl, vektor_lt, f"R + T {name}")

    return s_all, inliers


"""
Deprecated, only needed if Data is in binary format
"""


def convert_kitti_bin_to_pcd(bin, name):
    """
    Converts binary data to open3d point cloud
    :param bin: Numpy Array if Data is in binary format
    :param name: name of the data
    :return: open3d point cloud
    """
    size_float = 4
    list_pcd = []
    with open(bin, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            # if 4 <= x <= 30 and -5 <= y <= 5: # necessary if pre-cutting is wanted
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = toPointCloud(np_pcd)
    if debug:
        o3d.visualization.draw_geometries([pcd], height=800, width=800, mesh_show_back_face=False)
    return pcd


"""
Section 1: Support functions
"""


def transform_stereo(ob):
    """
    Transformates Stereo data to lidar coordinate systems
    :param ob: open3d point cloud
    :return: open3d point cloud transformed
    """
    trans_matrix = np.array([[0., -1., 0.],
                             [0., 0., -1.],
                             [1., 0., 0.]])
    np_object_isolated = np.array(ob.points)
    object1 = np.matmul(np_object_isolated, trans_matrix)
    object1 = toPointCloud(object1)
    return object1


def toPointCloud(points):
    """
    Converts Numpy Array to open3d point cloud
    :param points: numpy array
    :return: open3d point cloud
    """
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    return pointcloud


def remove_points_extended(file, cut):
    """
    Remove points outside of a defined boundary
    :param file: data in open3d point cloud format
    :param cut: boundarys with x, y, z
    :return: cropped data
    """
    point = np.asarray(file.points)
    point_new = point[(point[:, 0] > cut[0]) & (point[:, 0] < cut[1])
                      & (point[:, 1] > cut[2]) & (point[:, 1] < cut[3])
                      & (point[:, 2] > cut[4]) & (point[:, 2] < cut[5])]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def remove_points(file, i):
    """
    Remove points on the y-axle smaller than 0
    :param file: open3d point cloud
    :param i: threshold
    :return: cropped data
    """
    point = np.asarray(file.points)
    point_new = point[(point[:, 2] > i)]
    pcd_new = toPointCloud(point_new)
    return pcd_new


def plane_intersect(a, b):
    """
    calculate intersection points of planes
    :param a: numpy array, plane 1
    :param b: numpy array, plane 2
    :return: vector of line
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])
    aXb_vec = np.cross(a_vec, b_vec)

    return aXb_vec


def find_winkel(plane1, plane2, name):
    """
    calculate angle
    :param plane1: numpy array, plane 1
    :param plane2: numpy array, plane 2
    :param name: string, name of plane
    """
    plane1 = np.squeeze(np.asarray(plane1))
    plane2 = np.squeeze(np.asarray(plane2))
    nenner = np.dot(plane1[:3], plane2[:3])
    x_modulus = np.sqrt((plane1[:3] * plane1[:3]).sum())
    y_modulus = np.sqrt((plane2[:3] * plane2[:3]).sum())
    cos_angle = nenner / x_modulus / y_modulus
    angle = np.arccos(cos_angle)
    angle2 = angle * 360 / 2 / np.pi
    print(f"Winkel {name}:", angle2)


def geteuclideandistance(points_lidar, points_stereo):
    """
    Get euclidean distance of intersection points of lidar and stereo data
    :param points_lidar: numpy array, intersection points lidar
    :param points_stereo: numpy array, intersection points stereo
    """
    dist = []
    for i in range(len(points_lidar)):
        dist.append(distance.euclidean(points_lidar[i, :], points_stereo[i, :]))
    print("Euclidean Distance", dist)


def test_function(intersection, p1, p2, p3, name):
    """
    Tests intersections with all equations if equals 0, if not prints value
    :param intersection: numpy array, intersection
    :param p1: numpy array, plane 1
    :param p2: numpy array, plane 2
    :param p3: numpy array, plane 3
    :param name: name of plane
    """
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


def point_alignments(r, t, datapoints):
    """
    Aligns stereo data on lidar data with given rotation and translation
    :param datapoints: numpy array, data of stereo packet
    :param r: numpy array, rotation
    :param t: numpy array, translation
    :return: right side, left side, top side, intersection points, all transformed
    """
    points_aligned = np.add(np.dot(datapoints, r.T), t.T)
    return points_aligned


"""
Section 2: Algorithms
"""


def compute_distance(data, data_object, threshold, name):
    """
    Computes the distance between two point clouds and removes the points bigger then set threshold.
    In the second step a statistical outlier removal is applied (see next function)
    :param data: open3d point cloud without object
    :param data_object: open3d point cloud with object
    :param threshold: predefined threshold
    :param name: Stereo or Lidar
    :return: exposed object
    """
    dists = data_object.compute_point_cloud_distance(data)
    dists = np.asarray(dists)
    ind = np.where(dists > threshold)[0]
    dist_obj = data_object.select_by_index(ind)
    inlier_cloud = statistical_outlier(dist_obj, name)
    return inlier_cloud


def statistical_outlier(cloud, name):
    """
    Removes noise based on statistic
    :param cloud: open3d point cloud with object and noise
    :param name: name of the object
    :return: open3d point cloud object with reduced or no noise
    """
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=500, std_ratio=0.01)
    inlier_cloud = cloud.select_by_index(ind)
    if debug:
        display_inlier_outlier(cloud, ind, name)
    return inlier_cloud


def normal_estimation(downpcd):
    """
    Estimates normales of point cloud
    :param downpcd: open3d point cloud
    :return: open3d point cloud with normals
    """
    downpcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    downpcd.orient_normals_consistent_tangent_plane(200)
    if debug:
        showPointCloud(downpcd, "Normals", True)
    return downpcd


def kmeans(pc, name):
    """
    Applies k-Means Algorithm to object, with k-means+++
    :param pc: open3d point cloud
    :return: 3 planes: top, left, right as numpy array
    """
    normals = np.asarray(pc.normals)
    points = np.asarray(pc.points)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=1000, n_init=10)

    y_kmeans = kmeans.fit_predict(normals)
    # visualising the clusters
    if debug:
        centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                             kmeans.cluster_centers_[:, 2],
                             s=8, c='yellow', label='Centroids')

        t1 = getTrace(points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2], s=4, c='red',
                      label='Top')  # match with red=1 initial class
        t2 = getTrace(points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2], s=4, c='green',
                      label='Left')  # match with green=3 initial class
        t3 = getTrace(points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2], s=4, c='blue',
                      label='Right')  # match with blue=2 initial class

        showGraph(
            f"k-Means {name}",
            "Z", "X", "Y",
            [t1, t2, t3])  # , centroids])

    top_p = np.stack((points[y_kmeans == 0, 0], points[y_kmeans == 0, 1], points[y_kmeans == 0, 2]), axis=1)
    left_p = np.stack((points[y_kmeans == 1, 0], points[y_kmeans == 1, 1], points[y_kmeans == 1, 2]), axis=1)
    right_p = np.stack((points[y_kmeans == 2, 0], points[y_kmeans == 2, 1], points[y_kmeans == 2, 2]), axis=1)

    right_pc = toPointCloud(right_p)
    left_pc = toPointCloud(left_p)
    top_pc = toPointCloud(top_p)
    return right_pc, left_pc, top_pc


def ransac(plane, threshold, n, i):
    """
    Computes plane equation with ransac algorithm and gets side of given plane (top, right, left)
    :param plane: open3d point cloud
    :param threshold: threshold for plane
    :param n: number of start points
    :param i: number of iterations
    :return: inliers (numpy array), outliers (numpy array), plane equation (numpy array), name
    """
    plane_model, inliers = plane.segment_plane(distance_threshold=threshold,
                                               ransac_n=n,
                                               num_iterations=i)
    [a, b, c, d] = plane_model
    name = equation(a, b, c)
    # if debug:
    # print(f"Plane equation {name}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    inlier_cloud = plane.select_by_index(inliers)
    outlier_cloud = plane.select_by_index(inliers, invert=True)
    inlier_cloud = np.asarray(inlier_cloud.points)
    outlier_cloud = np.asarray(outlier_cloud.points)

    return inlier_cloud, outlier_cloud, np.array([[a, b, c, d]]), name  # np.array([[a, b, c, d]])


def icp(l_all, s_all):
    """
    Iterative closest point algorithm to find convergence between lidar and stereo data
    :param l_all: numpy array, intersection points lidar
    :param s_all: numpy array, intersection points stereo
    :return: rotation and translation matrices
    """
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


def intersection(equation, cm, intersect):
    """
    Find other intersection based on dimensions of object
    :param equation: numpy array, vector of intersection line
    :param cm: float, dimension of object in cm
    :param intersect: numpy array, start intersection
    :return: numpy array, calculated intersection point
    """
    a = np.multiply(equation, cm)
    s = np.subtract(intersect.reshape(-1, 3), a)
    s_shape = np.array(s)
    return s_shape


def equation(plane_x, plane_y, plane_z):
    """
    Gets side of given plane
    :param plane_x: numpy array, x-value of plane equation
    :param plane_y: numpy array, y-value of plane equation
    :param plane_z: numpy array, z-value of plane equation
    :return: plane name as string
    """
    plane_name = ""
    if plane_x > 0 and plane_y > 0 and plane_z > 0:
        plane_name = "Right"
    elif plane_x < 0 and plane_z > 0:
        plane_name = "Top"
    elif plane_x < 0 and plane_z < 0:
        plane_name = "Left"
    return plane_name


def finde_intersection(plane1, plane2, plane3, name):
    """
    Finds intersection points and test them against equations
    :param plane1: numpy array, first plane
    :param plane2: numpy array, second plane
    :param plane3: numpy array, third plane
    :param name: string, name of the plane
    :return: numpy array with intersection
    """
    left_side = [plane1[:3]] + [plane2[:3]] + [plane3[:3]]
    right_side = [[-plane1[3]]] + [[-plane2[3]]] + [[-plane3[3]]]
    i_p = np.linalg.solve(left_side, right_side)
    if debug:
        test_function(i_p, plane1, plane2, plane3, name)
    ip = np.array(i_p.reshape(-1, 3))
    return ip


"""
Section 3: Plot functions
"""


def display_inlier_outlier(cloud, ind, string):
    """
    Shows points point cloud with outlier points (red) and inlier points (grey) after statistical outlier removal
    :param cloud: open3d point cloud object
    :param ind: index if outlier or inlier
    :param string: name of object
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      window_name=string, height=800, width=800, mesh_show_back_face=False)


def getTrace(x, y, z, c, label, s=2):
    """
    Prepares data for plotting in plotly, accepts points in 3D-Coordinatesystem
    :param x: x-value in numpy array
    :param y: y-value in numpy array
    :param z: z-value in numpy array
    :param c: color of points
    :param label: label of points
    :param s: size of points
    :return: plotly trace points
    """
    trace_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label
    )
    return trace_points


def getMesh(x, y, z, c, label, v, s=4):
    """
    Prepares data for plotting in plotly, draws lines
    :param x: numpy array, x-value
    :param y: numpy array, y-value
    :param z: numpy array, z-value
    :param c: string, color of points
    :param label: string, label of points
    :param s: int, size of points
    :param v: string, visibility, set to only appear on legend
    :return: plotly line mesh
    """
    surface_points = go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
        name=label,
        visible=v
    )
    return surface_points


def showPointCloud(object, name, show_normal):
    """
    Plots point cloud
    :param object: open3d point cloud
    :param name: string, name ob plot
    :param show_normal: boolean, displays normals
    """
    if name == '':
        name = "Objekt"
    if show_normal == '':
        show_normal = False
    o3d.visualization.draw_geometries([object], "name", height=800, width=800,
                                      point_show_normal=show_normal, mesh_show_back_face=False)


def showGraph(title, x_colname, y_colname, z_colname, traces):
    """
    Shows plotly plot in browser
    :param title: string, headline of plot
    :param x_colname: string, name of x-axle
    :param y_colname: string, name of y-axle
    :param z_colname: string, name of z-axle
    :param traces: plotly format, data to display
    """
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


def drawIntersectionLines(intersections, name, color):
    """
    prepares data for plotly, namely plots intersection lines
    :param intersections: numpy array with all intersection points
    :param name: name of point cloud
    :param color: color of plotted lines
    :return: plotly trace lines
    """
    line_1 = np.concatenate(([intersections[0]], [intersections[2]], [intersections[4]],
                             [intersections[3]], [intersections[0]]), axis=0)
    line_2 = np.concatenate(([intersections[0]], [intersections[1]], [intersections[5]], [intersections[2]]), axis=0)
    line_3 = np.concatenate(([intersections[1]], [intersections[6]], [intersections[3]]), axis=0)

    line_1 = getMesh(line_1[:, 0], line_1[:, 1], line_1[:, 2],
                     c=color, label=f"{name}", v="legendonly")
    line_2 = getMesh(line_2[:, 0], line_2[:, 1], line_2[:, 2],
                     c=color, label=f"{name}", v="legendonly")
    line_3 = getMesh(line_3[:, 0], line_3[:, 1], line_3[:, 2],
                     c=color, label=f"{name}", v="legendonly")
    return line_1, line_2, line_3


if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Debug on/off")
    parser.add_argument("-dl", "--distance_lidar", help="Value for Distance Computing for Lidar",
                        default=0.1, type=float)
    parser.add_argument("-ds", "--distance_stereo", help="Value for Distance Computing for Stereo",
                        default=0.05, type=float)
    parser.add_argument("-rl", "--ransac_lidar", help="Value for Ransac Threshold for Lidar",
                        default=0.01, type=float)
    parser.add_argument("-rs", "--ransac_stereo", help="Value for Ransac Threshold for Stereo",
                        default=0.004, type=float)
    parser.add_argument("-cr", "--crop", help="Value for cropping the data in y-axle",
                        default=0.5, type=float)
    args = parser.parse_args()
    debug = args.debug
    threshold = args.crop
    """
    Lidar Data, have to be changed
    """
    lidar = "data/doppel_paket/seq_6.5m_styropor_pos1_0/lidar/1611244442.622.pcd", \
            "data/doppel_paket/seq_10m_styropor_pos1_0/lidar/merged.pcd", \
            "data/doppel_paket/seq_10m_styropor_pos2_0/lidar/merged.pcd", \
            "data/doppel_paket/seq_6.5m_empty_room_0/lidar/1611243759.606.pcd", \
            "data/doppel_paket/seq_6.5m_styropor_pos1_0/lidar/merged.pcd", \
            "data/doppel_paket/seq_6.5m_empty_room_0/lidar/merged.pcd"

    object_lidar = "data/doppel_paket/seq_6.5m_pos1_0/lidar/1611244398.218.pcd", \
                   "data/doppel_paket/seq_10m_pos1_0/lidar/merged.pcd", \
                   "data/doppel_paket/seq_10m_pos2_0/lidar/merged.pcd", \
                   "data/doppel_paket/seq_6.5m_pos2_0/lidar/1611244495.292.pcd", \
                   "data/doppel_paket/seq_6.5m_pos2_0/lidar/merged.pcd"

    """
    Stereo Data, have to be changed
    """
    stereo = "data/doppel_paket/seq_6.5m_styropor_pos1_0/stereo/1611244446.276.txt", \
             "data/doppel_paket/seq_10m_styropor_pos1_0/stereo/merged.txt", \
             "data/doppel_paket/seq_10m_styropor_pos2_0/stereo/merged.txt", \
             "data/doppel_paket/seq_6.5m_empty_room_0/stereo/1611243765.143.txt", \
             "data/doppel_paket/seq_6.5m_empty_room_0/stereo/merged.txt"
    object_stereo = "data/doppel_paket/seq_6.5m_pos1_0/stereo/1611244400.579.txt", \
                    "data/doppel_paket/seq_10m_pos1_0/stereo/merged.txt", \
                    "data/doppel_paket/seq_10m_pos2_0/stereo/merged.txt", \
                    "data/doppel_paket/seq_6.5m_pos2_0/stereo/1611244497.982.txt", \
                    "data/doppel_paket/seq_6.5m_pos2_0/stereo/merged.txt"

    """ Change value for using more than one data"""
    """"""
    v = 1
    """"""

    lidar = lidar[:v]
    object_lidar = object_lidar[:v]
    stereo = stereo[:v]
    object_stereo = object_stereo[:v]

    s_all_l = s_all_s = inliers_l = inliers_s = np.empty((0, 3))
    for i in range(len(lidar)):
        """
        Read Data
        """
        print("Read data from", lidar[i])
        file_lidar = o3d.io.read_point_cloud(lidar[i], format='auto')
        print("Read data from", object_lidar[i])
        file_object_lidar = o3d.io.read_point_cloud(object_lidar[i], format='auto')
        print("Read data from", stereo[i])
        file_stereo = o3d.io.read_point_cloud(stereo[i], format='xyzrgb')
        print("Read data from", object_stereo[i])
        file_object_stereo = o3d.io.read_point_cloud(object_stereo[i], format='xyzrgb')

        """ Cropping of data if necessary """
        crop_lidar = [5, 8, -1.5, 2, -0.5, 1]
        crop_lidar_10m = [10, 11.5, 0, 1, 0, 1]

        """ Threshold for cropping in y-axle
            -0.5 for 6.5m
            0 for 10m
        """
        if i >= 1:
            threshold = 0
        """
        remove_points can be changed to remove_points_extended for cropping of x, y and z axles
        """
        file_lidar = remove_points(file_lidar, threshold)
        file_object_lidar = remove_points(file_object_lidar, threshold)

        """
        Transform Stereodata to Lidar coordinate system
        """
        file_stereo_t = transform_stereo(file_stereo)
        file_object_stereo_t = transform_stereo(file_object_stereo)

        file_stereo_c = remove_points(file_stereo_t, threshold)
        file_object_stereo_c = remove_points(file_object_stereo_t, threshold)

        """ Run main on Lidar and Stereo """
        s_all_l_, inliers_l_ = main(file_lidar, file_object_lidar, args.distance_lidar,
                                    "Lidar", args.ransac_lidar)
        s_all_l = np.append(s_all_l, s_all_l_, axis=0)
        inliers_l = np.append(inliers_l, inliers_l_, axis=0)
        print("Lidar finished")
        s_all_s_, inliers_s_ = main(file_stereo_c, file_object_stereo_c, args.distance_stereo,
                                    "Stereo", args.ransac_stereo)
        s_all_s = np.append(s_all_s, s_all_s_, axis=0)
        inliers_s = np.append(inliers_s, inliers_s_, axis=0)
        print("Stereo finished")

    """
    Show point clouds before icp
    """
    if debug:
        inliers1_l = getTrace(inliers_l[:, 0], inliers_l[:, 1], inliers_l[:, 2], s=4, label="Lidar", c="green")
        inliers1_s = getTrace(inliers_s[:, 0], inliers_s[:, 1], inliers_s[:, 2], s=4, label="Stereo", c="orange")

        schnittpunkt1l = getTrace(s_all_l[:, 0], s_all_l[:, 1], s_all_l[:, 2], s=6, c='blue',
                                  label=f'S: Lidar')

        schnittpunkt1s = getTrace(s_all_s[:, 0], s_all_s[:, 1], s_all_s[:, 2], s=6, c='red',
                                  label=f'S: Stereo')
        showGraph(
            "Point Clouds",
            "Z", "X", "Y",
            [schnittpunkt1l, inliers1_l,
             schnittpunkt1s, inliers1_s])

    """ Compute center of mass and singular value decomposition """
    rotation, translation = icp(s_all_l, s_all_s)
    """ Allign Stereo data to lidar data """
    inliers_s = point_alignments(rotation, translation, inliers_s)
    s_all_s = point_alignments(rotation, translation, s_all_s)

    """ Get euclidean distance between intersections """
    geteuclideandistance(s_all_s, s_all_l)

    """
    Show point clouds after icp
    """
    if debug:
        l1, l2, l3 = drawIntersectionLines(s_all_l, "Lidar", "lightsteelblue")
        s1, s2, s3 = drawIntersectionLines(s_all_s, "Stereo", "salmon")
        """ Build trace for plotly """
        inliers_l = getTrace(inliers_l[:, 0], inliers_l[:, 1], inliers_l[:, 2], s=4, label="Lidar", c="green")
        inliers_s = getTrace(inliers_s[:, 0], inliers_s[:, 1], inliers_s[:, 2], s=4, label="Stereo", c="orange")

        schnittpunkt1l = getTrace(s_all_l[:, 0], s_all_l[:, 1], s_all_l[:, 2], s=6, c='blue',
                                  label=f'S: Lidar')

        schnittpunkt1s = getTrace(s_all_s[:, 0], s_all_s[:, 1], s_all_s[:, 2], s=6, c='red',
                                  label=f'S: Stereo')

        showGraph(
            "Point Clouds aligned",
            "Z", "X", "Y",
            [schnittpunkt1l, inliers_l,
             schnittpunkt1s, inliers_s,
             l1, l2, l3,
             s1, s2, s3])
    print("--- %s seconds ---" % (time.time() - start_time))
