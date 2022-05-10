from loguru import logger
import mmcv
import os
import os.path as osp
import time
import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import numpy as np
import argparse
from datetime import datetime



parser = argparse.ArgumentParser()
parser.add_argument('out_dir')
parser.add_argument('-n', default=1000, type=int)
parser.add_argument('--with_webcam', '-w', default=1000, type=int)
# parser.add_argument('--arguments', '-a', nargs='+', default=None)
args = parser.parse_args()


def save_thread_function(img, depth,intrinsic_mat, name):
    mmcv.imwrite(img, f'{name}.jpg')
    mmcv.dump(dict(depth=depth, intrinsic_mat=intrinsic_mat), f'{name}.pkl')

def vis_3d(rgbd):
    import open3d as o3d
    global geometry, vis

    rgb = rgbd[...,:3]/255.
    de = rgbd[..., 3:]
    # import ipdb; ipdb.set_trace()
    color = o3d.cpu.pybind.geometry.Image(rgb)
    de = 255/(de+1e-12)
    depth = o3d.cpu.pybind.geometry.Image(de)
    # depth_raw = o3d.io.read_image("../../test_data/RGBD/depth/00000.png")
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)#, 1, 50, False)

    # with open(args.intrinsic, 'r') as f:
    #     K = f.read().split()
    fx,cx = 594.90771484,239.74267578
    fy,cy = 594.90771484,319.23379517
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(height=480, width=640, fx=fx,fy=fy,cx=cx,cy=cy)
    extrinsic = np.diag(np.array([-1, -1, -1, 1]))
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic
    )
    o3d.visualization.draw(pcd)





class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

    def on_new_frame(self):
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # Notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self):
        print('Stream stopped')

    def connect_to_device(self, dev_idx):
        print('Searching for devices')
        devs = Record3DStream.get_connected_devices()
        print('{} device(s) found'.format(len(devs)))
        for dev in devs:
            print('\tID: {}\n\tUDID: {}\n'.format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError('Cannot connect to device #{}, try different index.'
                               .format(dev_idx))

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)  # Initiate connection and start capturing

    def get_intrinsic_mat_from_coeffs(self, coeffs):
        return np.array([[coeffs.fx,         0, coeffs.tx],
                         [        0, coeffs.fy, coeffs.ty],
                         [        0,         0,         1]])


    def start_processing_stream(self):
        # intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
        # print(intrinsic_mat)
        now = datetime.now()
        version = now.strftime("%b%d-%H:%M:%S")
        version = version.replace(':', '-')

        out_dir = osp.join(args.out_dir, version)
        i = 0

        cap = cv2.VideoCapture(0)
        pbar = mmcv.ProgressBar(args.n)
        import threading

        thread_list = []
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            ret, webcam_bgr = cap.read()

            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # import ipdb; ipdb.set_trace()
            webcam_bgr = mmcv.imrescale(webcam_bgr, (480, 480))
            # import ipdb; ipdb.set_trace()
            if args.with_webcam:
                img = np.concatenate([rgb, webcam_bgr], 0)
            else:
                img = rgb

            name = f'{out_dir}/{i:06d}'
            i+= 1

            # save_thread_function
            x = threading.Thread(target=save_thread_function, args=(img, depth, intrinsic_mat, name))
            x.start()            
            thread_list.append(x)
            self.event.clear()
            pbar.update()
            if i > args.n:
                break
        for _ in thread_list:
            _.join()
        logger.info('Finish')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
