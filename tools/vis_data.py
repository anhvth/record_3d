import numpy as np
from avcv.visualize import *
import mmcv
import open3d as o3d
import itertools
from glob import glob

fx,cx = 594.90771484,239.74267578
fy,cy = 594.90771484,319.23379517

def vis_3d(rgbd):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(height=480, width=640, fx=fx,fy=fy,cx=cx,cy=cy)
    extrinsic = np.diag(np.array([-1, -1, -1, 1]))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic
    )
    # o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw(pcd)




paths = glob('/Users/bi/gitprojects/record3d/data/May10-14-24-00/rgb/*.jpg')
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()

from tqdm import tqdm
for i, path in tqdm(enumerate(sorted(paths))):
    img = mmcv.imread(path, channel_order='rgb')[:640]
    # img.shape
    depth = mmcv.load(path.replace('/rgb/', '/depth/').replace('.jpg', '.pkl'))['depth']

    depth[np.isnan(depth)] = 0
    depth[depth>1.5] = 0
    # show(img)

    def convert_from_uvd(u, v, d):
        # d *= pxToMetre
        x_over_z = (cx - u) / fx
        y_over_z = (cy - v) / fy
        z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
        x = x_over_z * z
        y = y_over_z * z
        return x, y, z

    points = []
    rgb = []

    img = img/255.
    for v, u in itertools.product(range(640), range(480)):
        d = depth[v,u]
        if d > 0.1 and d <2:
            x,y,z = convert_from_uvd(u, v, d)    
            points.append([x,y,z])
            rgb.append(img[v,u])


    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if i == 0:
        vis.add_geometry(pcd)
    else:
        vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
vis.destroy_window()
