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

#============================================================================================
# web api


#============================================================================================
# TRAIN FEATURES
#============================================================================================
# receive a features' zip & train them:
#
# OPERATIONS
# get session & user IDs
# check, prepare file system and get submitted file
# => I create           instance/users_data/APIKEY/train_data/training_sessionid/data
# save zip 
# => I copy zip to      instance/users_data/APIKEY/train_data/training_sessionid/data.zip
# => I extract zip to   instance/users_data/APIKEY/train_data/training_sessionid/data
# copy json from        instance/..../training_sessionid/data => instance/..../training_sessionid/training.json
# read from it the nModelType
# add present session into the db
# start training :      train.train_net(session_data, session_path, session_uid, voicebank_vocabulary_path, True)
#
@training_api_blueprint.route('/api/v1/training-sessions', methods=['POST'])
def add_training_session():

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)

    # list of available commands
    voicebank_vocabulary_path = os.path.join(app.instance_path, 'inputnet', 'voicebank_commands.json')

    # get session & user IDs
    userkey = current_user.get_key()
    session_uid = uuid.uuid1()
    
    # check, prepare file system and get submitted file
    session_path = os.path.join(app.instance_path, 'users_data', userkey, 'train_data', str(session_uid))
    print(session_path)
    if os.path.exists(session_path):
        msg = 'ERROR: training session %d already exist' % session_uid
        raise Exception(msg)
    data_path = os.path.join(session_path, 'data')
    os.makedirs(data_path)
    file = request.files['file']

    # save zip & extract it
    filename = secure_filename(file.filename)
    file_path = os.path.join(session_path, 'data.zip')
    file.save(file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # copy json to ../ (train_data/training_sessionid), read it
    src_json_filename = os.path.join(data_path, 'training.json')
    dest_json_filename = os.path.join(session_path, 'training.json')
    os.rename(src_json_filename, dest_json_filename)

    with open(dest_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    # get nModelType from submitted json
    modeltype = session_data['nModelType']
    preproctype = session_data['nProcessingScheme']

    # add present session into the db
    training_session = TrainingSession(session_uid, modeltype, preproctype)
    if user_exists(current_user):
        training_session.user_id = current_user.id
    db.session.add(training_session)
    db.session.commit()

    # start training
    parallel_execution = True
    if parallel_execution:
        executor = ThreadPoolExecutor(2)
        executor.submit(train.train_net, session_data, session_path, session_uid, voicebank_vocabulary_path, True)
    else:
        train.train_net(session_data, session_path, session_uid, voicebank_vocabulary_path, True)

    return jsonify({'session_uid': session_uid}), 201, {'Location': training_session.get_url()}

#============================================================================================
# CHECK TRAIN COMPLETION
#============================================================================================
# returns:
# - 'status': 'pending' 
# or
# - json with the created net
#
@training_api_blueprint.route('/api/v1/training-sessions/<session_uid>', methods=['GET'])
def get_training_session(session_uid):
    
    #session_path = os.path.join('project', 'data', str(session_uid))
    session_path = os.path.join('project', 'data', str(session_uid))
    if (not os.path.exists(session_path)):
        abort(404)

    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()

    if training_session is None:
        abort(404)

    if not access_allowed(training_session, current_user):
        abort(401)

    if not training_session.completed:
        res = {'status': 'pending'}
        return jsonify(res)

    # training completed
    session_json_filename = os.path.join(session_path, 'training.json')
    with open(session_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    modeltype = session_data['nModelType']

    if modeltype == 274:
        trainparams_json = os.path.join('project', 'training_api', 'pure_user_trainparams.json')
    elif modeltype == 275:
        trainparams_json = os.path.join('project', 'training_api', 'pure_user_adapted_trainparams.json')    
    elif modeltype == 276:
        trainparams_json = os.path.join('project', 'training_api', 'common_adapted_trainparams.json')    
    elif modeltype == 277:
        trainparams_json = os.path.join('project', 'training_api', 'user_readapted_trainparams.json')  

    with open(trainparams_json, 'r') as data_file:
        train_data = json.load(data_file)

    nitems = len(session_data['commands'])

    output_net_name = "optimized_%s_%s_%d.pb" % (train_data['sModelFileName'], session_uid, session_data['nProcessingScheme'])

    # create return JSON
    # bLoaded, nDataDest, AssetManager are not sent back
    nw = datetime.now()
    res = {'status': 'complete',
           'sLabel': session_data['sLabel'],
           'nModelType': session_data['nModelType'],
           'nInputParams': train_data['nInputParams'],
           'nContextFrames': train_data['nContextFrames'],
           'nItems2Recognize': nitems,
           'sModelFileName': output_net_name,
           'sInputNodeName': train_data['sInputNodeName'],
           'sOutputNodeName': train_data['sOutputNodeName'],
           'fRecognitionThreshold': train_data['fRecognitionThreshold'],           
           'sLocalFolder': session_data['sLocalFolder'],
           'nProcessingScheme': session_data['nProcessingScheme'],
           'sCreationTime': nw.strftime('%Y/%m/%d %H:%M:%S'),
           'sessionid': str(session_uid),
           'commands': session_data['commands']
           }

    with open(session_json_filename, 'w') as data_file:
        json.dump(res, data_file)

    return jsonify(res)

#============================================================================================
# REQUEST NET TO DOWNLOAD
#============================================================================================
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

#============================================================================================
# accessory
#============================================================================================
def access_allowed(training_session, user):
    return training_session.user_id is None or (user_exists(user) and training_session.user_id == user.id)

def user_exists(user):
    return hasattr(user, 'id')
#============================================================================================