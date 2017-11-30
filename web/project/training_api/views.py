from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response, send_file
from flask_login import login_user, current_user, login_required, logout_user
from project import db, auth, auth_token, app, images
import uuid
import os
import zipfile
import json
from werkzeug import secure_filename
from project.exceptions import RequestException
from .libs import context, train
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from project.models import TrainingSession, User
from project import db, app

training_api_blueprint = Blueprint('training_api', __name__)

# the root path is : /home/flask/app/web

# I create          data/training_sessionid/data/matrices
# I copy zip to     data/training_sessionid/data.zip
# I unzip to        data/training_sessionid/data
# copy              data/training_sessionid/data/training.json => data/training_sessionid/training.json
#
# read from JSON the following:
# - processing scheme
# - commands list
# - modeltype (fine tune init net  OR  new net)


@training_api_blueprint.route('/api/v1/training-sessions', methods=['POST'])
def add_training_session():

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)

    session_uid = uuid.uuid1()
    print(session_uid)

    # check & prepare file system
    session_path = os.path.join('project', 'data', str(session_uid))
    if os.path.exists(session_path):
        msg = 'ERROR: training session %d already exist' % session_uid
        raise Exception(msg)
    file = request.files['file']

    data_path = os.path.join(session_path, 'data')
    os.makedirs(data_path)

    # save zip & extract it
    filename = secure_filename(file.filename)
    file_path = os.path.join(session_path, 'data.zip')
    file.save(file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # copy json to ../ (data/training_sessionid), read it
    src_json_filename = os.path.join(data_path, 'training.json')
    dest_json_filename = os.path.join(session_path, 'training.json')
    os.rename(src_json_filename, dest_json_filename)

    with open(dest_json_filename, 'r') as data_file:
        train_data = json.load(data_file)

    # newnet or ft_initnet
    modeltype = train_data['nModelType']
    # list of commands id [1102, 1103, 1206, ...]
    commands = train_data['commands']
    commands_ids = [cmd['id'] for cmd in commands]
    str_proc_scheme = str(train_data['nProcessingScheme'])  # 252/253/254/255

    training_session = TrainingSession(session_uid)
    if user_exists(current_user):
        training_session.user_id = current_user.id
    db.session.add(training_session)
    db.session.commit()

    parallel_execution = True

    if parallel_execution:
        executor = ThreadPoolExecutor(2)
        executor.submit(train.train_net, session_uid,
                    modeltype, commands_ids, str_proc_scheme, True)
    else:
        train.train_net(session_uid,
                    modeltype, commands_ids, str_proc_scheme, True)

    return jsonify({
        'training_session_id': session_uid
    })


@training_api_blueprint.route('/api/v1/training-sessions/<session_uid>', methods=['GET'])
def get_training_session(session_uid):
    session_path = os.path.join('project', 'data', str(session_uid))
    if (not os.path.exists(session_path)):
        abort(404)

    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()

    if not access_allowed(training_session, current_user):
        abort(401)

    if not training_session.completed:
        res = {'status': 'pending'}
        return jsonify(res)

    session_json_filename = os.path.join(session_path, 'training.json')
    with open(session_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    modeltype = session_data['nModelType']
    if modeltype == 274:
        trainparams_json = os.path.join(
            'project', 'training_api', 'train_params.json')
    else:
        trainparams_json = os.path.join(
            'project', 'training_api', 'ft_train_params.json')

    with open(trainparams_json, 'r') as data_file:
        train_data = json.load(data_file)

    nitems = len(session_data['commands'])

    output_net_name = "optimized_%s_%s_%d.pb" % (
        train_data['sModelFileName'], session_uid, session_data['nProcessingScheme'])

    # create return JSON
    nw = datetime.now()
    res = {'status': 'complete',
           'sLabel': session_data['sLabel'],
           'nModelType': session_data['nModelType'],
           'nInputParams': train_data['nInputParams'],
           'nContextFrames': train_data['nContextFrames'],
           'nItemsToRecognize': nitems,
           'sModelFileName': output_net_name,
           'sInputNodeName': train_data['sInputNodeName'],
           'sOutputNodeName': train_data['sOutputNodeName'],
           'sLocalFolder': session_data['sLocalFolder'],
           'nProcessingScheme': session_data['nProcessingScheme'],
           'sCreationTime': nw.strftime('%Y/%m/%d %H:%M:%S'),
           'commands': session_data['commands']
           }

    with open(session_json_filename, 'w') as data_file:
        json.dump(res, data_file)

    return jsonify(res)


@training_api_blueprint.route('/api/v1/training-sessions/<session_uid>/network', methods=['GET'])
def get_training_session_network(session_uid):
    directory_name = os.path.join(app.root_path, 'data', str(session_uid))
    if not os.path.isdir(directory_name):
        abort(404)

    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()

    if not access_allowed(training_session, current_user):
        abort(401)

    train_data_filepath = os.path.join(directory_name, 'training.json')
    with open(train_data_filepath, 'r') as train_data_file:
        train_data = json.load(train_data_file)
    filename = train_data['sModelFileName']
    print(filename)
    filepath = os.path.join(directory_name, filename)
        
    if (os.path.isfile(filepath)):
        attachment_filename = session_uid
        return send_file(filepath, attachment_filename=attachment_filename)
    else:
        abort(404)

def access_allowed(training_session, user):
    return training_session.user_id is None or (user_exists(user) and training_session.user_id == user.id)

def user_exists(user):
    return hasattr(user, 'id')