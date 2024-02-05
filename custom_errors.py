class FaceDetectionError(Exception):

    def __init__(self, message: str = "Problem in face detection"):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message


class FaceMatchingError(Exception):

    def __init__(self, message: str = "Problem with Face Matching"):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return self.message