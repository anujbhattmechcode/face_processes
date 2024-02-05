from API import facial_process_api
from API.views import *


facial_process_api.add_url_rule(
    '/ping', view_func=ping
)
facial_process_api.add_url_rule(
    '/crop', view_func=FaceCropper.as_view('face_crop')
)
facial_process_api.add_url_rule(
    '/match', view_func=FaceMatching.as_view('face_matching')
)