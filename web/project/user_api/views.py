from flask import render_template, Blueprint, request, redirect, url_for, flash, abort, jsonify
from sqlalchemy.exc import IntegrityError
from flask_login import login_user, current_user, login_required, logout_user
from threading import Thread
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime

from project import db, app
from project.models import User, Device

user_api_blueprint = Blueprint('user_api', __name__)

@user_api_blueprint.route('/api/v1/api_key_reset', methods=["POST"])
def api_key_reset():
    if not user_exists(current_user):
        abort(401)
    current_user.refresh_login()
    current_user.regenerate_api_key()
    db.session.add(current_user)
    db.session.commit()
    res = {'api_key': current_user.api_key}
    return jsonify(res)

@user_api_blueprint.route('/api/v1/devices/<string:uuid>', methods=["GET"])
def get_device(uuid):
    if not user_exists(current_user):
        abort(401)
    device = Device.query.filter_by(uuid = uuid).first()
    if device is None:
        abort(404)
    if device.user_id != current_user.id:
        abort(401)
    return jsonify(device.export_data())

@user_api_blueprint.route('/api/v1/devices', methods=["POST"])
def new_device():
    if not user_exists(current_user):
        abort(401)
    json_data = request.get_json()
    uuid = json_data['uuid']
    device = Device.query.filter_by(uuid = uuid).first()
    if device is None:
        device = Device()
    device.import_data(current_user, request)
    db.session.add(device)
    db.session.commit()
    return jsonify({}), 201, {'Location': device.get_url()}

def user_exists(user):
    return hasattr(user, 'id')