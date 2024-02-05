import base64

import numpy as np
import cv2 as cv

from dataclasses import dataclass


@dataclass
class Holder:
    """
    Data holder class, store and transform attributes
    """
    uuid: str = None
    output: str = None
    cropped_face: np.ndarray = None
    detected_face: np.ndarray = None
    message: str = None
    confidence: float = None

    def to_dict(self) -> dict:
        """
        Converts the attributes to dict, numpy image objects are converted to base64 object to return as response
        :return: dict object
        """
        output_dict = {}

        if self.uuid:
            output_dict["uuid"] = self.uuid
        if isinstance(self.cropped_face, np.ndarray):
            output_dict["cropped_face"] = Holder.convert_to_base64(self.cropped_face)
        if isinstance(self.detected_face, np.ndarray):
            output_dict["detected_face"] = Holder.convert_to_base64(self.detected_face)
        if self.output:
            output_dict["output"] = self.output
        if self.message:
            output_dict["message"] = self.message
        if self.confidence:
            output_dict["confidence"] = self.confidence

        return output_dict

    @staticmethod
    def convert_to_base64(im: np.ndarray) -> str:
        """
        Converts the numpy array image to bytes
        :param im: (np.ndarray) numpy object image
        :return: str object after converting image to base64
        """
        _, img_encoded = cv.imencode('.png', im)
        img_base64 = base64.b64encode(
            img_encoded
        ).decode('utf-8')

        return img_base64
