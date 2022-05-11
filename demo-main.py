import os
import os.path as osp
import time
import numpy as np
from record3d import Record3DStream
import cv2
from threading import Event
import numpy as np
import argparse
parser = argparse.ArgumentParser()
    # import cv2
import mediapipe as mp



mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh




def lmk2np(lmks):
    res = []
    for lmk in lmks:
        res.append([lmk.x, lmk.y, lmk.z])
    return np.array(res)


def mediapipe_facemesh_predict(image):

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()
        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return annotated_image
        for face_landmarks in results.multi_face_landmarks:
            # lmk_np = lmk2np(face_landmarks.landmark)
            # np.save(f'{args.out_np_dir}/{idx:06d}', lmk_np)
            # if args.vis:
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
    return annotated_image







parser.add_argument('--dump', action='store_true', default=False)
parser.add_argument('--vis3d','-v', action='store_true', default=False)

parser.add_argument('--rgbd_dir', default='data/rgbd/')
parser.add_argument('--max_d', default=1.2, type=float)
args = parser.parse_args()

if args.vis3d:
    from open3d_utils import Visualizer3D
    vis3d = Visualizer3D()


if args.dump:
    os.makedirs(args.rgbd_dir, exist_ok=True)
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
        while True:
            self.event.wait()  # Wait for new frame to arrive

            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()
            intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())
            # You can now e.g. create point cloud by projecting the depth map using the intrinsic matrix.

            # Postprocess it
            # import ipdb; ipdb.set_trace()
            depth[np.isnan(depth)] = 0
            depth[depth>args.max_d] = 0
            # depth = 1/(depth+1e-6)
            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Show the RGBD Stream

            # import ipdb; ipdb.set_trace()
            if args.vis3d:
                vis3d.visualize(rgb[...,::-1], depth)
                
            else:
                rgb = np.rot90(rgb)
                depth = np.rot90(depth)
                rgb = mediapipe_facemesh_predict(rgb)
                cv2.imshow('RGB', rgb)
                cv2.imshow('Depth', depth)
                cv2.waitKey(1)
            self.event.clear()
        #== End loop
        # vis.destroy_window()


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
