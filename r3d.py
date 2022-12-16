import numpy as np
import open3d as o3d
import numpy as np
import cv2
import liblzfse  # https://pypi.org/project/pyliblzfse/
from convert import depth_map_to_point_cloud
from avcv.all import *

def load_depth(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img

def load_conf(filepath):
    with open(filepath, 'rb') as depth_fh:
        raw_bytes = depth_fh.read()
        # print(raw_bytes)
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)

    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img
# @memoizem
def load_r3d(depth_filepath, meta=json.load(open('./data/r3d/metadata')), min_conf=2):
    depth_img = load_depth(depth_filepath).copy()
    rgb = cv2.imread(depth_filepath.replace('.depth', '.jpg'))[...,::-1].copy()
    conf = load_conf(depth_filepath.replace('.depth', '.conf'))
    depth_img[conf<min_conf] = 0
    depth_img = mmcv.imrescale(depth_img, (meta['w'], meta['h']), interpolation='nearest')
    conf = mmcv.imrescale(conf, (meta['w'], meta['h']), interpolation='nearest')
    xyz = depth_map_to_point_cloud(depth_img, meta['h'], meta['w']).reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)/255.
    return rgb, xyz, conf


if __name__ == '__main__':
    from avcv.all import *
    paths = sorted(glob('./data/r3d/rgbd/*.depth'), key=lambda path:int(get_name(path)))
    meta = json.load(open('./data/r3d/metadata'))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    # multi_process(load_r3d, paths)
    for i, path in enumerate(paths):
        rgb, xyz, conf = load_r3d(path)
        conf = conf.reshape(-1)
        rgb = rgb[conf>=2]
        xyz = xyz[conf>=2]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        if i == 0:
            vis.add_geometry(pcd)
        else:
            vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()