from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response
from project import db, auth, auth_token, app, images
import uuid, os
from werkzeug import secure_filename
from project.exceptions import RequestException
from .libs import context, train
from concurrent.futures import ThreadPoolExecutor

training_api_blueprint = Blueprint('training_api', __name__)

@training_api_blueprint.route('/users/<int:user_id>/training-sessions', methods=['POST'])
def add_training_session(user_id):

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)
    
    training_sessionid = uuid.uuid1()
    print(training_sessionid)

    folder_path = os.path.join('data', str(training_sessionid))
    if os.path.exists(folder_path): 
        msg = 'ERROR: training session %d already exist' % training_sessionid
        raise Exception(msg)

    file = request.files['file']

    print ('create dir')
    os.makedirs(folder_path)

    filename = secure_filename(file.filename)
    file_path = os.path.join(folder_path, 'data.zip')
    file.save(file_path)

    executor = ThreadPoolExecutor(2)

    executor.submit(train.train_net, training_sessionid, user_id)
    # train.train_net(training_sessionid, user_id)

    return jsonify({'training_session_id': training_sessionid})

@training_api_blueprint.route('/users/<int:user_id>/training-sessions/<session_id>', methods=['GET'])
def get_training_session(user_id, session_id):
    folder_path = os.path.join('data', str(session_id))
    if (not os.path.exists(folder_path)):
        abort(404)
    lockfile_path = os.path.join(folder_path, '.lock')
    if (os.path.exists(lockfile_path)):
        res = {'status': 'pending'}
        return jsonify(res)
    
    res = {'status': 'complete'}
    return jsonify(res)
    




