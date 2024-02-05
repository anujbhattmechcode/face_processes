"""
Define all the view function here
"""
import numpy as np
from flask import request
from flask.views import MethodView

from API import pipeline_obj
from helpers import base64_to_numpy
from custom_errors import *


class FaceMatching(MethodView):
    """
    Class based views for face mathing of the given two images
    """
    def post(self):
        """
        Runs core model pipeline to match the two faces
        """
        try:
            im1 = request.files.get("image_1")
            im2 = request.files.get("image_2")

            if not im1:
                return {"BAD_REQUEST": "image_1 is corrupted or none"}, 400
            if not im2:
                return {"BAD_REQUEST": "image_2 is corrupted or none"}, 400

            try:
                im1 = im1.read()
                im1 = base64_to_numpy(im1)
            except Exception as E:
                return {
                    "BAD_REQUEST": "image_1 corrupted or not image file",
                    "ERROR": E.__str__()
                }, 400

            if not isinstance(im1, np.ndarray):
                return {
                    "BAD_REQUEST": "image_1: not an image file"
                }, 400

            try:
                im2 = im2.read()
                im2 = base64_to_numpy(im2)
            except Exception as E:
                return {
                    "BAD_REQUEST": "image_2 corrupted or not image file",
                    "ERROR": E.__str__()
                }, 400

            if not isinstance(im2, np.ndarray):
                return {
                    "BAD_REQUEST": "image_2: not an image file"
                }, 400

            output = pipeline_obj.face_matching(im1, im2)
        except Exception as E:
            return {"CRITICAL": "INTERNAL SERVER ERROR",
                    "MESSAGE": E.__str__()}, 500

        return output


class FaceCropper(MethodView):
    """
    Class based views for face detection and cropping
    """
    def post(self):
        """
        Runs core model pipeline to detect and crop face
        """
        try:
            im = request.files.get("image")
            if not im:
                return {"BAD_REQUEST": "image is corrupted or none"}, 400

            try:
                im = im.read()
                im = base64_to_numpy(im)

            except Exception as E:
                return {
                    "BAD_REQUEST": "image corrupted or not image file"
                }, 400

            if not isinstance(im, np.ndarray):
                return {
                    "BAD_REQUEST": "not an image file"
                }, 400

            output = pipeline_obj.crop_face(im)

            return output

        except Exception as E:
            return {"CRITICAL": "INTERNAL SERVER ERROR",
                    "MESSAGE": E.__str__()}, 500


def ping():
    """
    Ping function to check the status of the API
    """
    return {"STATUS": "API IS LIVE"}
