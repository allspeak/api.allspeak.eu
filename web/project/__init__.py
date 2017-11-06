#################
#### imports ####
#################

from os.path import join, isfile
import os

from flask import Flask, render_template, make_response, jsonify, send_file, request, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_pagedown import PageDown
from flask_migrate import Migrate
from flask_httpauth import HTTPBasicAuth
import requests
import traceback
from .exceptions import RequestException


################
#### config ####
################

app = Flask(__name__, instance_relative_config=True)

app.config.from_pyfile('flask.cfg')


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
pagedown = PageDown(app)
migrate = Migrate(app, db)
auth = HTTPBasicAuth()
auth_token = HTTPBasicAuth()

# Configure the image uploading via Flask-Uploads
images = UploadSet('images', IMAGES)
configure_uploads(app, images)


####################
#### blueprints ####
####################

from project.training_api.views import training_api_blueprint

# register the blueprints
app.register_blueprint(training_api_blueprint)

############################
#### custom error pages ####
############################

@app.errorhandler(RequestException)
def exception(e):
    print(str(e))
    return make_response(jsonify({'error': str(e)}), 400)

@app.errorhandler(Exception)
def exception(e):
    print(str(e))
    traceback.print_exc()
    return make_response(jsonify({'error': str(e)}), 500)

@app.errorhandler(400)
def request_error(e):
    return make_response(jsonify({'error': 'Wrong request'}), 400)

@app.errorhandler(404)
def not_found(e):
    return make_response(jsonify({'error': 'Required URL does not exist'}), 404)


@app.errorhandler(403)
def page_not_found(e):
    return make_response(jsonify({'error': 'Forbidden'}), 403)

@app.errorhandler(401)
def request_error(e):
    return make_response(jsonify({'error': 'Forbidden'}), 401)