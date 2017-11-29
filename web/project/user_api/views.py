from flask import render_template, Blueprint, request, redirect, url_for, flash, abort, jsonify
from sqlalchemy.exc import IntegrityError
from flask_login import login_user, current_user, login_required, logout_user
from threading import Thread
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime

from project import db, app
from project.models import User

user_api_blueprint = Blueprint('user_api', __name__)

@user_api_blueprint.route('/api/v1/api_key_reset', methods=["POST"])
def api_key_reset():
    current_user.refresh_login()
    current_user.regenerate_api_key()
    db.session.add(current_user)
    db.session.commit()
    res = {'api_key': current_user.api_key}
    return jsonify(res)