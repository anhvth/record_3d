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
    
class FaceModel:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.mp_drawing = mp.solutions.drawing_utils
        self.draw_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # self.draw_spec = self.mp_face_mesh.DrawingSpec(thickness=1, circle_radius=1)
    def __call__(self, img, draw=False):
        results = self.face_mesh.process(img)
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

hmodel = HumanSegmentation()
fmodel = FaceModel()