# from open3d.geometry.PointCloud import compute_point_cloud_distance
import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("no_person.ply")
    # vol = o3d.visualization.read_selection_polygon_volume(
    #     "no_person.ply")
    vol = o3d.io.read_point_cloud("person.ply")
    # chair = vol.crop_point_cloud(pcd)

    dists = pcd.compute_point_cloud_distance(vol)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.5)[0]
    pcd_without_chair = pcd.select_by_index(ind)
    o3d.visualization.draw_geometries([pcd_without_chair])



