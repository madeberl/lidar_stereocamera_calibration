import os
from tkinter.filedialog import askopenfilename
from tkinter import Tk
import numpy as np
import open3d as o3d
import struct


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
    o3d.visualization.draw_geometries([object], window_name=str(i),
                                          height=800, width=800)
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
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=200))
    return downpcd


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
    o3d.visualization.draw_geometries([normals_estimated], "Normale", height=800, width=800,
                                      point_show_normal=True)
    o3d.io.write_point_cloud("packet_data/test_data/packet_normalised.ply", normals_estimated)
