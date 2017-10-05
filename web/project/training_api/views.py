from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response
from project import db, auth, auth_token, app, images
import json, uuid, os, zipfile
from werkzeug import secure_filename
from project.exceptions import RequestException

training_api_blueprint = Blueprint('training_api', __name__)

@training_api_blueprint.route('/users/<int:user_id>/training-sessions', methods=['POST'])
def add_training_session(user_id):
    training_sessionid = uuid.uuid1()
    folder_path = os.path.join('data', str(training_sessionid))
    if os.path.exists(folder_path): 
        msg = 'ERROR: training session %d already exist' % training_sessionid
        raise Exception(msg)

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)
    
    file = request.files['file']

    os.makedirs(folder_path)

    filename = secure_filename(file.filename)
    file_path = os.path.join(folder_path, 'data.zip')
    file.save(file_path)
    
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)

    return jsonify({'training_session_id': training_sessionid})