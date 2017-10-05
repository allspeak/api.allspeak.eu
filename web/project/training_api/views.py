from flask import render_template, Blueprint, request, redirect, url_for, abort, jsonify, g, make_response
from project import db, auth, auth_token, app, images
import json, uuid, os, zipfile
from werkzeug import secure_filename
from project.exceptions import RequestException
from .libs import context, train
from concurrent.futures import ThreadPoolExecutor

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

    json_filename = os.path.join(folder_path, 'training.json')

    with open(json_filename, 'r') as data_file:
        train_data = json.load(data_file)

    ctx_frames = train_data['nContextFrames']    # num contexting frames
    vocabulary = train_data['vocabulary']     # list of commands id [1102, 1103, 1206, ...]
    commands_ids = [cmd['id'] for cmd in vocabulary]

    inputnet_path = 'project/inputnet/optimized_allcontrols_fsc.pb'
    output_net_name = "user_%d_FT_net_fsc" % user_id
    output_net_path = os.path.join(folder_path, 'net')

    # model params
    ftnet_vars_name = "fine_tuning_weights"
    batch_size = 100
    hm_epochs = 20       # NUMBER OF CYCLES: feed forward + backpropagation
    clean_folder = True
    # CONTEXTING DATA (create ctx_...  files)
    # context.createSubjectContext(folder_path, ctx_frames)

    executor = ThreadPoolExecutor(2)

    executor.submit(train.train_net, folder_path, ctx_frames, commands_ids, inputnet_path, output_net_path, output_net_name, clean_folder, hm_epochs, batch_size, ftnet_vars_name)

    return jsonify({'training_session_id': training_sessionid})