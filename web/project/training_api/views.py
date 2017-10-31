from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response
from project import db, auth, auth_token, app, images
import uuid, os
import zipfile
import json
from werkzeug import secure_filename
from project.exceptions import RequestException
from .libs import context, train
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
 
training_api_blueprint = Blueprint('training_api', __name__)


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

    # check & prepare file system
    session_path = os.path.join('data', str(training_sessionid))
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

    # copy json to ../ (data/training_sessionid), read it 
    src_json_filename = os.path.join(data_path, 'training.json')
    dest_json_filename = os.path.join(session_path, 'training.json')
    os.rename(src_json_filename, dest_json_filename)

    with open(dest_json_filename, 'r') as data_file:
        train_data = json.load(data_file)

    modeltype = train_data['nModelType']    # newnet or ft_initnet
    vocabulary = train_data['vocabulary']     # list of commands id [1102, 1103, 1206, ...]
    commands_ids = [cmd['id'] for cmd in vocabulary]
    str_proc_scheme = str(train_data['nProcessingScheme'])  # 252/253/254/255

    executor = ThreadPoolExecutor(2)
    executor.submit(train.train_net, training_sessionid, user_id, modeltype, commands_ids, str_proc_scheme)

    return jsonify({'training_session_id': training_sessionid})

@training_api_blueprint.route('/users/<int:user_id>/training-sessions/<session_id>', methods=['GET'])
def get_training_session(user_id, session_id):
    session_path = os.path.join('data', str(session_id))
    if (not os.path.exists(session_path)):
        abort(404)
    lockfile_path = os.path.join(session_path, '.lock')
    if (os.path.exists(lockfile_path)):
        res = {'status': 'pending'}
        return jsonify(res)

  
    session_json_filename = os.path.join(session_path, 'training.json')
    with open(session_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    modeltype = session_data['nModelType']
    if modeltype == 274:
        trainparams_json = 'training_api/train_params.json'
    else:
        trainparams_json = 'training_api/ft_train_params.json'
    
    with open(trainparams_json, 'r') as data_file:
        train_data = json.load(data_file)

    nitems = len(session_data['vocabulary'])

    output_net_name = "optimized_%s_%d_%d.pb" % (train_data['sModelFileName'], user_id, session_data['nProcessingScheme'])

    # create return JSON  
    nw = datetime.now()
    res = { 'status': 'complete',
            'nModelType':session_data['nModelType'],
            'nInputParams':train_data['nInputParams'],
            'nContextFrames':train_data['nContextFrames'],
            'nItemsToRecognize':nitems,
            'sModelFilePath':output_net_name,
            'sInputNodeName':train_data['sInputNodeName'],
            'sOutputNodeName':train_data['sOutputNodeName'],            
            'nProcessingScheme':session_data['nProcessingScheme'],
            'sCreation':nw.strftime('%Y/%m/%d %H:%M:%S'),
            'vocabulary':session_data['vocabulary']
            }
    return jsonify(res)



@training_api_blueprint.route('/users/<int:user_id>/training-sessions/<session_id>/network', methods=['GET'])
def get_training_session_network(user_id, session_id):
    try:
        filename = "user_%d_FT_net_fsc" % user_id
        filepath = os.path.join('data', str(session_id), 'net', filename)
        if (os.path.isfile(filepath)):
            attachment_filename = session_id
            return send_file(filepath, attachment_filename=attachment_filename)
        else:
            abort(404)
    except Exception as e:
        print(str(e))
        abort(500)
