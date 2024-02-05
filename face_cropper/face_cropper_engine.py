import numpy as np
import settings as setting
import cv2 as cv

from custom_errors import *
from mtcnn.mtcnn import MTCNN


class FaceDetector:
    """
    Main class that detects and crops the face(s) from the given image
    """
    def __init__(self) -> None:
        self.face_detection = MTCNN()

    def inference(self, im: np.ndarray) -> dict:
        """
        Does facial recognition of passed image with images available in the dataset, it uses MTCNN for face
        detection and FaceNet for face recognition
        :param im: (np.ndarray) Image in numpy array format
        :return: (dict) output dict with the following
        face_detection_confidence -> (float) Face detection confidence
        face -> (np.ndarray) Cropped face
        detected_face -> (np.ndarray) Original image with detected face
        """
        if not isinstance(im, np.ndarray):
            raise TypeError("Image needs to be numpy array")

        face_detected = self.detect_face(im)

        return face_detected

    def detect_face(self, im: np.ndarray) -> dict:
        """
        Detects faces
        :param im: (np.ndarray) Input image
        :return: (dict) Output directory with the following values
        face_detection_confidence -> (float) Face detection confidence
        face -> (np.ndarray) Cropped face
        detected_face -> (np.ndarray) Original image with detected face
        """
        output = {}
        detected_face = self.face_detection.detect_faces(im)
        if len(detected_face) > 1:
            raise FaceDetectionError("Input image has more than one face")

        if len(detected_face) == 0:
            raise FaceDetectionError("No faces found")

        output["face_detection_confidence"] = float(detected_face[0]["confidence"])
        if output["face_detection_confidence"] < setting.FACE_DETECTION_THRESHOLD:
            raise FaceDetectionError(f"Low face detection confidence")

        x, y, w, h = detected_face[0]['box']

        output["face"] = im[y:y + h, x: x + w]
        output["detected_face"] = cv.rectangle(
            im, (x, y), (x + w,y + h), (0, 255, 255), 2
        )

        return output
