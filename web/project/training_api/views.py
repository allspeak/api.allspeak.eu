from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response, send_file
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


@training_api_blueprint.route('/users/<int:user_id>/training-sessions', methods=['POST'])
def add_training_session(user_id):

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)

    training_sessionid = uuid.uuid1()
    print(training_sessionid)
    auth_token = uuid.uuid1()

    # check & prepare file system
    session_path = os.path.join('project', 'data', str(training_sessionid))
    if os.path.exists(session_path):
        msg = 'ERROR: training session %d already exist' % training_sessionid
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

    auth_filepath = os.path.join(session_path, 'auth')
    f = open(auth_filepath, 'w')
    f.write(str(auth_token))
    f.close()

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

    executor = ThreadPoolExecutor(2)
    executor.submit(train.train_net, training_sessionid, user_id,
                    modeltype, commands_ids, str_proc_scheme, True)
    return jsonify({
        'training_session_id': training_sessionid,
        'auth_token': auth_token
    })


@training_api_blueprint.route('/users/<int:user_id>/training-sessions/<session_id>', methods=['GET'])
def get_training_session(user_id, session_id):
    session_path = os.path.join('project', 'data', str(session_id))
    if (not os.path.exists(session_path)):
        abort(404)

    auth_filepath = os.path.join(session_path, 'auth')
    auth_token_in = request.headers.get('auth_token')
    f = open(auth_filepath, 'r')
    auth_token = f.read()
    f.close()
    if auth_token_in != auth_token:
        abort(401)

    lockfile_path = os.path.join(session_path, '.lock')
    if (os.path.exists(lockfile_path)):
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

    output_net_name = "optimized_%s_%d_%d.pb" % (
        train_data['sModelFileName'], user_id, session_data['nProcessingScheme'])

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


@training_api_blueprint.route('/users/<int:user_id>/training-sessions/<session_id>/network', methods=['GET'])
def get_training_session_network(user_id, session_id):
    directory_name = os.path.join(app.root_path, 'data', str(session_id))
    if not os.path.isdir(directory_name):
        abort(404)
        

    auth_filepath = os.path.join(directory_name, 'auth')
    auth_token_in = request.headers.get('auth_token')
    f = open(auth_filepath, 'r')
    auth_token = f.read()
    f.close()
    if auth_token_in != auth_token:
        abort(401)

    train_data_filepath = os.path.join(directory_name, 'training.json')
    with open(train_data_filepath, 'r') as train_data_file:
        train_data = json.load(train_data_file)
    filename = train_data['sModelFileName']
    filepath = os.path.join(directory_name, filename)
    if (os.path.isfile(filepath)):
        attachment_filename = session_id
        return send_file(filepath, attachment_filename=attachment_filename)
    else:
        abort(404)

# @training_api_blueprint.route('/users/guest/training-sessions', methods=['POST'])