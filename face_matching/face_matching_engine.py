import numpy as np
import settings as setting

# from keras_vggface.vggface import VGGFace
from deepface import DeepFace


class FaceMatchingEngine:
    """
    Main class that does face matching
    """
    def __init__(self) -> None:
        # self.face_recognition = VGGFace(
        #     model='resnet50',
        #     include_top=False,
        #     input_shape=(224, 224, 3),
        #     pooling='avg'
        # )
        self.face_recognition = DeepFace

    def inference(self, im1: np.ndarray, im2: np.ndarray) -> dict:
        """
        Does facial recognition of im1 with im2
        :param im1: (np.ndarray) First image
        :param im2: (np.ndarray) First image

        :return: (dict) output dict with the following
        match -> MATCH or NO_MATCH
        confidence -> confidence of the result
        """
        if not isinstance(im1, np.ndarray):
            raise TypeError("Image needs to be numpy array")

        if not isinstance(im2, np.ndarray):
            raise TypeError("Image needs to be numpy array")

        face_recognizer = self.match_faces(im1, im2)

        return face_recognizer

    def face_recognition(self, im_source: np.ndarray, im_target: np.ndarray) -> dict:
        """
        Recognizes face against given face dataset
        :param im_source: (np.ndarray) First image
        :param im_target: (np.ndarray) First image
        :return: (dict) Output directory with the following values
        match -> MATCH or NO_MATCH
        confidence -> confidence of the result
        """
        im_source_emb = self.embedding(im_source)
        im_target_emb = self.embedding(im_target)

        similarity = FaceMatchingEngine.cosine(im_source_emb, im_target_emb)

        if similarity < setting.FACE_MATCHING_THRESHOLD:
            msg = "NO_MATCH"
        else:
            msg = "MATCH"

        return {
            "match": msg,
            "confidence": similarity
        }

    def match_faces(self, im_source: np.ndarray, im_target: np.ndarray) -> dict:
        """
        Recognizes face against given face dataset
        :param im_source: (np.ndarray) First image
        :param im_target: (np.ndarray) First image
        :return: (dict) Output directory with the following values
        match -> MATCH or NO_MATCH
        confidence -> confidence of the result
        """
        face_recognizer = self.face_recognition.verify(im_source, im_target, enforce_detection=False)
        similarity = 1 - face_recognizer["threshold"]

        if similarity < setting.FACE_MATCHING_THRESHOLD:
            msg = "NO_MATCH"
        else:
            msg = "MATCH"

        return {
            "match": msg,
            "confidence": similarity
        }

    @staticmethod
    def cosine(v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def embedding(self, image: np.ndarray):
        """
        Generates face embedding for the given face
        :param image: (np.ndarray)  image
        :return:
        """
        image /= 255.0
        image = (image - image.mean()) / image.std()

        embedding = self.face_recognition.predict(np.expand_dims(image, axis=0))

        return embedding
