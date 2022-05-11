import itertools
import time
from glob import glob

import mmcv
import sys
import os
import os.path as osp
sys.path.insert(0, osp.dirname(__file__)+'/../')
from open3d_utils import Visualizer3D


paths = glob('/Users/bi/gitprojects/record3d/data/May10-14-24-00/rgb/*.jpg')


from tqdm import tqdm

list_points =  []
list_rgb = []
pbar = mmcv.ProgressBar(len(paths))




vis3d = Visualizer3D()
for i, path in tqdm(enumerate(sorted(paths))):
    img = mmcv.imread(path, channel_order='rgb')[:640]
    depth = mmcv.load(path.replace('/rgb/', '/depth/').replace('.jpg', '.pkl'))['depth']
    vis3d.visualize(img, depth)
    time.sleep(0.2)
vis3d.close()




