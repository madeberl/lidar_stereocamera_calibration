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
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    o3d.io.write_point_cloud(name + ".ply", pcd)
    return pcd


def compute_distance(data, data_person):
    dists = data.compute_point_cloud_distance(data_person)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.01)[0]
    data_without_person = data.select_by_index(ind)
    return data_without_person


if __name__ == "__main__":
    # file = "2011_09_26_drive_0001_sync/velodyne_points/data/0000000000.bin"
    file = "no_person.bin"
    file_person = "person.bin"
    name = file.split(".")
    name2 = file_person.split(".")
    pcd = convert_kitti_bin_to_pcd(file, name[0])
    pcd_person = convert_kitti_bin_to_pcd(file_person, name2[0])
    # distance = compute_distance(pcd, pcd_person)
    o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_geometries([pcd_person])
