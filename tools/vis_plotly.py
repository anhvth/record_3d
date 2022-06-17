import numpy as np
import time
from glob import glob
import open3d as o3d
import mmcv

def get_points_from_img_and_depth(depth, img=None, max_d=1):
    pcd = o3d.geometry.PointCloud()        
    depth[np.isnan(depth)] = 0
    depth[depth>max_d] = 0

    vv,uu = np.where(np.logical_and(depth>0., depth<max_d))
    dd = depth[vv,uu]

    x,y,z = convert_from_uvd(uu, vv, dd, self.cx, self.cy, self.fx, self.fy)
    points = np.stack([x,z,y], 1)
    points = o3d.utility.Vector3dVector(points)
    rgb = img[vv,uu]/255.
    pcd.points = points
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd

def read_pcd_from_npy(path):
    np_point = np.load(path)
    pcd = o3d.geometry.PointCloud()        
    np_point = np_point.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(np_point)
    return pcd

def read_pcd_from_pkl(path):
    depth = mmcv.load(path)['depth']
    pcd = get_points_from_img_and_depth(depth)
    return pcd

class Visualizer3D:
    def __init__(self, fx=594.90771484,cx=239.74267578, fy=594.90771484,cy=319.23379517, min_d=0, max_d=100):
        self.fx,self.cx,self.fy,self.cy = fx,cx,fy,cy
        self.min_d = min_d
        self.max_d = max_d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        
        self.counter = 0

    def visualize(self, path):
        if path.endswith('npy'):
            pcd = read_pcd_from_npy(path)
        elif path.endswith('pkl'):
            pcd = read_pcd_from_pkl(path)
        else:
            pcd = o3d.io.read_point_cloud(path)
        self.pcd.points = pcd.points
        if self.counter == 0:
            self.vis.add_geometry(self.pcd)
        else:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.counter += 1

    def close(self):
        self.vis.destroy_window()

if __name__ == '__main__':
    import argparse
    import os
    import os.path as osp

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--ext', default='ply')
    parser.add_argument('--sleep', '-s', type=float, default=1/30)
    args = parser.parse_args()
    

    query_str = osp.join(args.input_dir, f'*.{args.ext}')
    print((query_str))
    paths = glob(query_str)
    paths = list(sorted(paths))

    assert len(paths), len(paths)
    if args.ext == 'pkl':
        from open3d_utils import Visualizer3D
        
    vis3d = Visualizer3D()
    for path in paths:
        vis3d.visualize(path)
        time.sleep(args.sleep)