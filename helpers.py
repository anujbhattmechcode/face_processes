import base64

import numpy as np
import cv2 as cv


def base64_to_numpy(im_base64):
    """
    Convert base64 encoding to NumPy array
    """
    im = cv.imdecode(np.frombuffer(im_base64, np.uint8), 3)

    return im
