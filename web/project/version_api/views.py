from flask import render_template, Blueprint, request, redirect, url_for, flash, abort, jsonify
from sqlalchemy.exc import IntegrityError
from flask_login import login_user, current_user, login_required, logout_user
from threading import Thread
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime

from project import db, app
from project.models import User, Device

version_api_blueprint = Blueprint('version_api', __name__)

# these api can be called by any user. 
# versions are stored in the 'versions' folder and are named  'allspeak_123.apk'
# versions info (nversion, sversion, description, update date) are stored in a db

#============================================================================================
# to upload a new version, user must be registered...I should check that user is also an admin
# admin submit the following information
# nversion = 123, sversion = '0.1.0', description = 'bla, bla bla bla......'
@version_api_blueprint.route('/api/v1/upload-version', methods=['POST'])
def add_new_version():

    if not user_exists(current_user):
        abort(401)

    if 'file' not in request.files or request.files['file'].filename == '':
        msg = 'ERROR: no file in request'
        raise RequestException(msg)

    nw = datetime.now()

    json_data = request.get_json()
    nversion = json_data['nversion']
    sversion = json_data['sversion']
    description = json_data['description']
    date = nw.strftime('%Y/%m/%d %H:%M:%S')

    # check & prepare file system

    filename = 'allspeak_' + str(nversion) + '.apk'    # non sono se sono stringhe

    filepath = os.path.join('project', 'versions', filename)
    if os.path.exists(filepath):
        msg = 'ERROR: new version file %d already exist' % nversion
        raise Exception(msg)
    file = request.files['file']


    # save zip & extract it
    #filename = secure_filename(file.filename)   # FEDE nell'upload della rete, non usi questa funzione in realtà, come mai??
    file.save(filepath)

    # insert info into the DB

    # training_session = TrainingSession(session_uid)
    # if user_exists(current_user):
    #     training_session.user_id = current_user.id
    # db.session.add(training_session)
    # db.session.commit()
    return jsonify({}), 201, {'Location': training_session.get_url()}   # non so cosa mettere in location



#============================================================================================
#receive a version number [as integer] (the App one), compares it with the last one, return 0 if updated, return last version suffix if not updated
@version_api_blueprint.route('/api/v1/exist-new-version/<string:nversion>', methods=["GET"])
def exist_new_version(nversion):
 
    # get last versiontraining_session
    last_nversion = 123
    if nversion < last_version:
        return str(last_version)
    else:
        return 0

#============================================================================================
# download a specific version, given a version id suffix
@version_api_blueprint.route('/api/v1/get-version/<version_id>', methods=['GET'])
def get_version(version_id):

    filename = 'allspeak_' + version_id + '.apk'
    filepath = os.path.join(app.root_path, 'versions', filename)
    print(filename)
        
    if (os.path.isfile(filepath)):
        return send_file(filepath, attachment_filename=filename)
    else:
        abort(404)


def user_exists(user):
    return hasattr(user, 'id')