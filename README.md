### Face Proccesses ###
Deep learning model powered various face processes can be done by the model.
Flask API is used along with deep learning models and libraries to achieve the tasks

Currently, it supports two operation: Face Cropping and Face Matching.

The entire project's configuration is provided in settings.py.
```python
# Minimum input image size
MINIMUM_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Face detection threshold
FACE_DETECTION_THRESHOLD: float = 0.5

```

Modify this file to make changes in threshold or other things.

### Flask API ###
The processes can be run via flask API.

Run the following command to run flask API:
```bash
foo@bar:~$ python go_live.py
```
This will run the API on 5010 port, you can change the port and other things by modifying go_live.py file

### Package ###
This processes can be used directly as well by using pipeline.py file.
```python
from pipeline import Pipeline
import cv2 as cv

face_processes = Pipeline()

im = cv.imread("face image")
im_cropped = face_processes.crop_face(im)

im1 = cv.imread("face image one")
im2 = cv.imread("face image two")
matcher = face_processes.face_matching(im1, im2)
```
