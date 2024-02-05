import uuid

import numpy as np

from holder import Holder
from typing import Tuple
from face_matching.face_matching_engine import FaceMatchingEngine
from face_cropper.face_cropper_engine import FaceDetector
from custom_errors import *


class Pipeline:
    """
    Main class that integrates all the available modules
    """
    def __init__(self):
        self.face_cropper = FaceDetector()
        self.face_match = FaceMatchingEngine()

    def crop_face(self, im: np.ndarray) -> Tuple[dict, int]:
        """
        Detects and crops the face
        :param im: (np.ndarray) Image
        :return: Tuple with dictionary and http status code
        """
        holder = Holder()
        holder.uuid = str(uuid.uuid4())
        try:
            face_match_output = self.face_cropper.inference(im)

        except FaceDetectionError as E:
            holder.message = E.__str__()
            return holder.to_dict(), 406

        holder.detected_face = face_match_output["detected_face"]
        holder.cropped_face = face_match_output["face"]
        holder.confidence = face_match_output["face_detection_confidence"]

        return holder.to_dict(), 200

    def face_matching(self, im1: np.ndarray, im2: np.ndarray) -> Tuple[dict, int]:
        """
        Does facial recognition of im1 with im2
        :param im1: (np.ndarray) First image
        :param im2: (np.ndarray) First image

        :return: Tuple with dictionary and http status code
        """
        holder = Holder()
        holder.uuid = str(uuid.uuid4())
        try:
            fc1 = self.face_cropper.inference(im1)["face"]

        except FaceDetectionError as E:
            holder.message = f'image_1: {E.__str__()}'
            return holder.to_dict(), 406

        try:
            fc2 = self.face_cropper.inference(im2)["face"]

        except FaceDetectionError as E:
            holder.message = f'image_2: {E.__str__()}'
            return holder.to_dict(), 406

        try:
            face_match = self.face_match.inference(fc1, fc2)
        except FaceMatchingError as E:
            holder.message = E.__str__()
            return holder.to_dict(), 406

        holder.confidence = face_match["confidence"]
        holder.message = face_match["match"]

        return holder.to_dict(), 200
