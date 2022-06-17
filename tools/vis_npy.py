# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    args = parser.parse_args()

    # pcd = o3d.io.read_point_cloud(args.input)
    np_point = np.load(args.input)


    pcd = o3d.geometry.PointCloud()        
    np_point = np_point.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(np_point)

    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
