# API initializer
from flask import Flask
from pipeline import Pipeline


facial_process_api = Flask(__name__)
pipeline_obj = Pipeline()

import API.url_rules
