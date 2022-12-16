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
import mediapipe as mp
import cv2



from open3d_utils import Visualizer3D
vis3d = Visualizer3D()

class HumanSegmentation:
    def __init__(self):
        # Load mediapipe selfie segmentation model.
        self.selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.selfie_segmentation.SelfieSegmentation(model_selection=1)
    def __call__(self, img):
        results = self.segmentation.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(img.shape, dtype=np.uint8)
        output_image = np.where(condition, img, bg_image)
        return output_image


# Create a FaceModel class, that use mediapipe to detect face landmarks. the __call__ function should take an image and return the visulized image with landmarks.
class FaceModel:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # self.draw_spec = self.mp_face_mesh.DrawingSpec(thickness=1, circle_radius=1)
    def __call__(self, img, draw=False):
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        annotated_image = img.copy()
        face_landmarks_np = None
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_landmarks_np = self.convert_face_landmarks_to_numpy_array(face_landmarks)
                if draw:

                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles
                        .get_default_face_mesh_contours_style())

                    # self.mp_drawing.draw_landmarks(
                    #     image=annotated_image,
                    #     landmark_list=face_landmarks,
                    #     connections=self.mp_face_mesh.FACEMESH_IRISES,
                    #     landmark_drawing_spec=None,
                    #     connection_drawing_spec=self.mp_drawing_styles
                    #     .get_default_face_mesh_iris_connections_style())

        return annotated_image, face_landmarks_np

    def convert_face_landmarks_to_numpy_array(self, face_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])


    def get_eye_contours(self, face_landmarks_np):
        left_eye = face_landmarks_np[33: 42]
        right_eye = face_landmarks_np[133: 142]
        return left_eye, right_eye

def rescale_padding(img, size):
    """
    Resize the image to the given size and pad it with zeros to the given size.
    """

    img = mmcv.imresize(img, size)
    img = mmcv.impad(img, shape=size, pad_val=0)
    return img

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


def pixel_coordinate_to_camera_coordinate(x, y, depth, intrinsic):
    """
    Convert pixel coordinate to camera coordinate.
    Args:
        x (int): x coordinate in pixel.
        y (int): y coordinate in pixel.
        depth (float): depth value.
        intrinsic (np.ndarray): camera intrinsic matrix.
    Returns:
        np.ndarray: 3D coordinate in camera coordinate.
    """
    fx, fy, cx, cy = intrinsic
    x = (x - cx) * depth / fx
    y = (y - cy) * depth / fy
    return np.array([x, y, depth])




class DemoApp:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.face_model = FaceModel()

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
        
        # self.body_segmentation = HumanSegmentation()

        cap = cv2.VideoCapture(0)
        human_model = HumanSegmentation()
        while True:
            self.event.wait()  # Wait for new frame to arrive
            # Copy the newly arrived RGBD frame
            depth = self.session.get_depth_frame()
            rgb = self.session.get_rgb_frame()

            depth = cv2.resize(depth, (480, 640))
            rgb = cv2.resize(rgb, (480, 640))


            depth = depth.T

            depth = depth[::-1]
            

            fore_ground_mask = (depth > 0) & (depth < 2)
            rgb = np.transpose(rgb, [1,0,2])
            rgb = rgb[::-1]


            h, w = depth.shape[:2]
            rgb = cv2.resize(rgb, (w, h))
            rgb = rgb * fore_ground_mask[..., None]


            # intrinsic_mat = self.get_intrinsic_mat_from_coeffs(self.session.get_intrinsic_mat())

            if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
                depth = cv2.flip(depth, 1)
                rgb = cv2.flip(rgb, 1)

            
            body_mask = human_model(rgb)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            body_mask= body_mask==0
            body_mask = (body_mask*255).astype(np.uint8)
            cv2.imshow('body_mask', body_mask)
            

            # depth[body_mask==0] = 0
            # import ipdb; ipdb.set_trace()
            # Detect and draw facemesh
            # rgb, face_landmarks_np = self.face_model(rgb, draw=False)
            # if face_landmarks_np is not None:
            #     # Convert face_landmarks_np to image coordinates
            #     face_landmarks_np = face_landmarks_np * np.array([w, h, 1])
            #     face_landmarks_np = face_landmarks_np.astype(np.int32)
            #     # Eye contours
            #     # Mediapipe left eye land mark: 36-41
            #     # Mediapipe right eye land mark: 42-47

            #     def draw_contour(img, points, color, thickness=1):
            #         # Draw contour
            #         points_convex = cv2.convexHull(points)
            #         cv2.polylines(img, [points_convex], True, color, thickness)


            #     def get_mean_depth(depth, points):
            #         points_convex = cv2.convexHull(points)
            #         # Create a zero mask and fill the convex hull with 1
            #         mask = np.zeros(depth.shape, dtype=np.uint8)
            #         cv2.fillConvexPoly(mask, points_convex, 1)
            #         # Get the mean depth of the convex hull
            #         mean_depth = np.mean(depth[mask == 1])
            #         return mean_depth

            #     left_eye = face_landmarks_np[[130, 243, 23, 27], :2]

            #     right_eye = face_landmarks_np[[359, 463, 253, 257], :2]
            #     # import ipdb; ipdb.set_trace()
            #     draw_contour(rgb, left_eye, (0, 0, 255), 1)
            #     draw_contour(rgb, right_eye, (0, 0, 255), 1)

            #     mean_depth_left_eye = get_mean_depth(depth, left_eye)
            #     # mean_depth_left_eye is the distance to the camera in meters, we can use it to calculate the 3d position of the eye
            #     # The 3d position of the eye is the center of the eye contour
            #     # Intrinsic matrix is the camera matrix
            #     # array([[434.9118042 ,   0.        , 240.229599  ],
            #     #     [  0.        , 434.9118042 , 321.36166382],
            #     #     [  0.        ,   0.        ,   1.        ]])
            #     intrinsic_mat = fx, fy, cx, cy = 434.9118042, 434.9118042, 240.229599, 321.36166382
            #     # import ipdb; ipdb.set_trace()

            #     left_eye_x = (left_eye[0][0] + left_eye[1][0]) / 2
            #     left_eye_y = (left_eye[0][1] + left_eye[1][1]) / 2
            #     eye_3d_left = pixel_coordinate_to_camera_coordinate(left_eye_x, left_eye_y, mean_depth_left_eye,intrinsic_mat)
            #     # Convert eye_3d_left from meters to centimeters
            #     eye_3d_left = [_ * 100 for _ in eye_3d_left]

            #     mean_depth_right_eye = get_mean_depth(depth, right_eye)
            #     right_eye_x = (right_eye[0][0] + right_eye[1][0]) / 2
            #     right_eye_y = (right_eye[0][1] + right_eye[1][1]) / 2
            #     eye_3d_right = pixel_coordinate_to_camera_coordinate(right_eye_x, right_eye_y, mean_depth_right_eye, intrinsic_mat)
            #     # Convert eye_3d_right from meters to centimeters
            #     eye_3d_right = [_ * 100 for _ in eye_3d_right]

                


            #     cv2.putText(rgb, 'Left eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(eye_3d_left[0], eye_3d_left[1], eye_3d_left[2]),
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     cv2.putText(rgb, 'Right eye 3d: {:.4f}, {:.4f}, {:.4f}'.format(eye_3d_right[0], eye_3d_right[1], eye_3d_right[2]),
            #                 (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


            #     # Put text mean depth to the image
            #     cv2.putText(rgb, 'Left eye mean depth: {:.2f}'.format(mean_depth_left_eye),
            #                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            #     cv2.putText(rgb, 'Right eye mean depth: {:.2f}'.format(mean_depth_right_eye),
            #                 (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                

            vis3d.visualize(rgb[...,::-1], depth)

            # Imshow
            if False:
                cv2.imshow('rgb', rgb)
                cv2.imshow('depth', depth)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self.event.clear()

        logger.info('Finish all')
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = DemoApp()
    app.connect_to_device(dev_idx=0)
    app.start_processing_stream()
