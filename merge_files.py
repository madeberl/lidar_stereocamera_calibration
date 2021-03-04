import os
import open3d as o3d
import numpy as np
from tkinter.filedialog import askdirectory
from tkinter import Tk

if __name__ == "__main__":
    Tk().withdraw()
    path = askdirectory(initialdir=os.getcwd(), title="Folder")
    dirListing = os.listdir('.')
    xyz = np.empty((0, 3))
    subdirs = [x[0] for x in os.walk(path)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        os.chdir(subdir)
        if len(files) > 0:
            r = []
            for file in files:
                if file.endswith(".pcd"):
                    pcd = o3d.io.read_point_cloud(file)
                    points = np.asarray(pcd.points)
                    xyz = np.append(xyz, points, axis=0)
                if file.endswith(".txt"):
                    r.append(file)
            if file.endswith(".pcd"):
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(xyz)
                o3d.io.write_point_cloud("merged.pcd", point_cloud)
                print(f"{subdir} closed")
            if file.endswith(".txt"):
                with open('merged.txt', 'w') as outfile:
                    for fname in r:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)
                print(f"{subdir} closed")
                outfile.close()





    # editFiles = []
    # xyz = np.empty((0, 3))
    # for item in dirListing:
    #     if item != 'merged.pcd':
    #         editFiles.append(item)
    # for file in editFiles:
    #     pcd = o3d.io.read_point_cloud(file)
    #     points = np.asarray(pcd.points)
    #     xyz = np.append(xyz, points, axis=0)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(xyz)
    # o3d.io.write_point_cloud("merged.pcd", point_cloud)
