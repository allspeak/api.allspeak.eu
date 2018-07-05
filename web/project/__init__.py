#################
#### imports ####
#################

from os.path import join, isfile
import os

from flask import Flask, render_template, make_response, jsonify, send_file, request, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_pagedown import PageDown
from flask_migrate import Migrate
from flask_httpauth import HTTPBasicAuth
import requests
import traceback
from .exceptions import RequestException, RequestExceptionPlus
from flask_cors import CORS, cross_origin


################
#### config ####
################

app = Flask(__name__, instance_relative_config=True)

app.config.from_pyfile('flask.cfg')
CORS(app)
def format_datetime(value):
    format="%Y-%m-%d %H:%M:%S"
    if value is None:
        return 'Never'
    else:
        return value.strftime(format)

app.jinja_env.filters['datetime'] = format_datetime

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

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "user.login"

# Configure the image uploading via Flask-Uploads
images = UploadSet('images', IMAGES)
configure_uploads(app, images)

from project.models import User

@login_manager.user_loader
def load_user(user_id):
    return User.query.filter(User.id == int(user_id)).first()


@login_manager.request_loader
def load_user_from_request(request):
    api_key = request.headers.get('api-key')
    if api_key is None:
        return None
    user = User.query.filter(User.api_key == api_key).first()
    return user


####################
#### blueprints ####
####################

from project.training_api.views import training_api_blueprint
from project.user.views import user_blueprint
from project.user_api.views import user_api_blueprint
from project.admin.views import admin_blueprint

# register the blueprints
app.register_blueprint(training_api_blueprint)
app.register_blueprint(user_blueprint)
app.register_blueprint(user_api_blueprint)
app.register_blueprint(admin_blueprint)

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

@app.errorhandler(RequestExceptionPlus)
def exceptionplus(e):
    print(str(e.get_msg1()))
    traceback.print_exc()
    return make_response(jsonify({'error': e.get_msg1(), 'ex': e.get_msg2()}), 500)

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