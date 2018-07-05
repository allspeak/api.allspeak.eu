from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response, send_file
from flask_login import login_user, current_user, login_required, logout_user
from project import db, auth, auth_token, app, images
import uuid
import os
import zipfile
import json
import shutil
from werkzeug import secure_filename
from project.exceptions import RequestException, RequestExceptionPlus
from .libs import context, train
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from project.models import TrainingSession, User, Error
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
# copy json from        instance/..../training_sessionid/data => instance/..../training_sessionid/vocabulary.json
# read from it the nModelType
# add present session into the db
# start training :      train.train_net(session_data, session_path, session_uid, voicebank_vocabulary_path, True)
#
@training_api_blueprint.route('/api/v1/training-sessions', methods=['POST'])
def add_training_session():

    cdir = os.getcwd()
    param_file1 = os.path.join('project', 'training_api', 'params', 'ff_pure_user_trainparams.json')
    param_file2 = os.path.join('web', 'project', 'training_api', 'params', 'ff_pure_user_trainparams.json')
    return jsonify({'param_file1': param_file1, 'param_file2': param_file2, 'cdir':cdir, 'res':os.path.exists(param_file)}), 201, {'Location': training_session.get_url()}

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)

    # list of available commands
    voicebank_vocabulary_path = os.path.join(app.instance_path, 'voicebank_commands.json')

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

    if not os.path.exists(data_path):
        msg = 'ERROR: training session data folder (%s) could not be created. Check your permissions' % data_path
        raise Exception(msg)        

    file = request.files['file']

    # save zip & extract it
    filename = secure_filename(file.filename)
    file_path = os.path.join(session_path, 'data.zip')
    file.save(file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_path)

    # copy json to users_data/USERKEY/train_data/training_sessionid, read it
    src_json_filename = os.path.join(data_path, 'vocabulary.json')
    dest_json_filename = os.path.join(session_path, 'vocabulary.json')

    if not os.path.isfile(src_json_filename):
        msg = 'ERROR: input json file is not present.'
        raise Exception(msg)      
    
    os.rename(src_json_filename, dest_json_filename)

    with open(dest_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    # get nModelType from submitted json
    nModelType = session_data['nModelType']
    preproctype = session_data['nProcessingScheme']

    if user_exists(current_user):
        user_id = current_user.id
    else:
        user_id = None

    try:
        # add present session into the db
        training_session = TrainingSession(session_uid, nModelType, preproctype)
        if user_id is not None:
            training_session.user_id = user_id
        db.session.add(training_session)
        db.session.commit()

    except Exception as e:
        print(e)
        raise RequestExceptionPlus(str(e), str(session_uid)) 

    # start training
    parallel_execution = True
    if parallel_execution:
        executor = ThreadPoolExecutor(2)
        executor.submit(train.train_net, session_data, session_path, session_uid, voicebank_vocabulary_path, user_id, True)
    else:
        train.train_net(session_data, session_path, session_uid, voicebank_vocabulary_path, user_id, True)

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
    
    # session_path = os.path.join('project', 'data', str(session_uid))
    # if (not os.path.exists(session_path)):
    #     abort(404)
    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()

    if training_session is None:
        abort(404)

    if not access_allowed(training_session, current_user):
        abort(401)

    if not training_session.completed:
        error = Error.query.filter_by(session_uid=session_uid).first()
        if error is None:
            res = {'status': 'pending'}
        else:
            res = {'status': 'error', 'description': error.description}
        return jsonify(res)

    # training completed
    net_file_path = training_session.net_path
    net_folder = os.path.dirname(net_file_path)
    session_json_filename = os.path.join(net_folder, 'vocabulary.json')
    with open(session_json_filename, 'r') as data_file:
        session_data = json.load(data_file)

    nModelType = session_data['nModelType']
    nModelClass = session_data['nModelClass']

    model_root_path = os.path.join('project', 'training_api', 'params')

    if nModelClass == 280:
        if nModelType == 274:
            trainparams_json = os.path.join(model_root_path, 'ff_pure_user_trainparams.json')
        elif nModelType == 275:
            trainparams_json = os.path.join(model_root_path, 'ff_pure_user_adapted_trainparams.json')    
        elif nModelType == 276:
            trainparams_json = os.path.join(model_root_path, 'ff_common_adapted_trainparams.json')    
        elif nModelType == 277:
            trainparams_json = os.path.join(model_root_path, 'ff_user_readapted_trainparams.json')  
        elif nModelType == 278:
            trainparams_json = os.path.join(model_root_path, 'ff_common_readapted_trainparams.json')  
    else:
        if nModelType == 274:
            trainparams_json = os.path.join(model_root_path, 'lstm_pure_user_trainparams.json')
        elif nModelType == 275:
            trainparams_json = os.path.join(model_root_path, 'lstm_pure_user_adapted_trainparams.json')    
        elif nModelType == 276:
            trainparams_json = os.path.join(model_root_path, 'lstm_common_adapted_trainparams.json')    
        elif nModelType == 277:
            trainparams_json = os.path.join(model_root_path, 'lstm_user_readapted_trainparams.json')  
        elif nModelType == 278:
            trainparams_json = os.path.join(model_root_path, 'lstm_common_readapted_trainparams.json')  

    with open(trainparams_json, 'r') as data_file:
        train_data = json.load(data_file)

    nitems = len(session_data['commands'])

    output_net_name = "%s_%s_%d_%d" % (train_data['sModelFileName'], str(nModelType), session_data['nProcessingScheme'], nModelClass)

    # create return JSON
    # bLoaded, nDataDest, AssetManager are not sent back
    nw = datetime.now()
    res = {'status': 'complete',
           'sLabel': session_data['sLabel'],
           'nModelClass': nModelClass,
           'nModelType': nModelType,
           'nInputParams': train_data['nInputParams'],
           'nContextFrames': train_data['nContextFrames'],
           'nItems2Recognize': nitems,
           'sModelFileName': output_net_name,
           'saInputNodeName': train_data['saInputNodeName'],
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

    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()
    net_path = training_session.net_path

    if not access_allowed(training_session, current_user):
        abort(401)

    if not os.path.isfile(net_path):
        abort(404)    
    else:
        attachment_filename = session_uid
        return send_file(net_path, attachment_filename=attachment_filename)


#============================================================================================
# DELETE TRAINING SESSION AND/OR UPLOADED FILES
#============================================================================================
@training_api_blueprint.route('/api/v1/training-sessions/<session_uid>/delete', methods=['GET'])
def delete_training_session(session_uid):

    # get training session folder & delete it
    userkey = current_user.get_key()
    session_path = os.path.join(app.instance_path, 'users_data', userkey, 'train_data', str(session_uid))
    if os.path.isdir(session_path):
       shutil.rmtree(session_path)

    # if a corresponding training session exist => remove it 
    # it may have in fact uploaded the session file but crashed during db interaction
    training_session = TrainingSession.query.filter_by(session_uid=session_uid).first()

    if training_session is not None:

        if not access_allowed(training_session, current_user):
            abort(401)

        # delete db entry
        db.session.delete(training_session)
        db.session.commit()

    print("training session : " + session_uid + " removed")

    return jsonify({'status': 'ok'})





#============================================================================================
# accessory
#============================================================================================
def access_allowed(training_session, user):
    return training_session.user_id is None or (user_exists(user) and training_session.user_id == user.id)

def user_exists(user):
    return hasattr(user, 'id')
#============================================================================================