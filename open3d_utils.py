import numpy as np
import open3d as o3d


def convert_from_uvd(u, v, d, cx, cy, fx,fy):
    # d *= pxToMetre
    x_over_z = (cx - u) / fx
    y_over_z = (cy - v) / fy
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z
    return x, y, z



class Visualizer3D:
    def __init__(self, fx=594.90771484,cx=239.74267578, fy=594.90771484,cy=319.23379517, min_d=0, max_d=100):
        self.fx,self.cx,self.fy,self.cy = fx,cx,fy,cy
        self.min_d = min_d
        self.max_d = max_d
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.pcd = o3d.geometry.PointCloud()
        
        self.counter = 0

    def visualize(self, img, depth):
        depth[np.isnan(depth)] = 0
        depth[depth>self.max_d] = 0

        vv,uu = np.where(np.logical_and(depth>0., depth<self.max_d))
        dd = depth[vv,uu]

        x,y,z = convert_from_uvd(uu, vv, dd, self.cx, self.cy, self.fx, self.fy)
        points = np.stack([x,z,y], 1)
        points = o3d.utility.Vector3dVector(points)
        rgb = img[vv,uu]/255.

        
        self.pcd.points = points
        self.pcd.colors = o3d.utility.Vector3dVector(rgb)
        if self.counter == 0:
            self.vis.add_geometry(self.pcd)
        else:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.counter += 1

    def close(self):
        self.vis.destroy_window()
