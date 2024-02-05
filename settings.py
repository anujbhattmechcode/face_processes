from typing import Literal, Tuple


# Minimum input image size
MINIMUM_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Face detection threshold
FACE_DETECTION_THRESHOLD: float = 0.5

"""
Database backend, where to store the data
If selected mongo, it makes uses mongo database with configuration provided at MONGO_CONFIG,
else, if selected local, it stores the images in local
"""
DATABASE_BACKEND: Literal["mongo", "local"] = "local"
MONGO_CONFIG = {
    "URI": "mongodb://0.0.0.0:9010",
    "DB_NAME": "FACE_LIBRARY",
    "COLL_NAME": "DATA"
}

# Face matching threshold
FACE_MATCHING_THRESHOLD = 0.5