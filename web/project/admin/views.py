from flask import render_template, Blueprint, request, redirect, url_for, flash, abort, jsonify, send_file, Response
from sqlalchemy.exc import IntegrityError
from flask_login import login_user, current_user, login_required, logout_user
from threading import Thread
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime
import os
from .forms import MobileApplicationForm
from project import db, app
from project.models import MobileApplication, User, PsysuiteApplication
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict

admin_blueprint = Blueprint('admin', __name__)


################
#### routes ####
################


@admin_blueprint.route('/mobile_application/new', methods=['GET', 'POST'])
@login_required
def mobile_application_new():
    if current_user.role != User.ADMIN:
        return redirect(url_for('user.index'))

    form = MobileApplicationForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                mobile_application = MobileApplication()
                mobile_application.version = form.version.data
                mobile_application.sver = form.sver.data
                mobile_application.description = form.description.data
                filename = 'allspeak%d.apk' % mobile_application.version
                filepath = os.path.join(app.instance_path, 'mobile_applications', filename)
                mobile_application.apk_path = filepath
                f = form.apk.data
                f.save(filepath)
                db.session.add(mobile_application)
                db.session.commit()
                flash('New mobile application added')
                return redirect(url_for('admin.mobile_application_latest'))
            except IntegrityError:
                db.session.rollback()
                flash('New mobile application added', 'error')
    mobile_application = MobileApplication.query.order_by(MobileApplication.id.desc()).first()
    if mobile_application:
        form.version.data = mobile_application.version + 1
        form.sver.data = mobile_application.sver
    return render_template('mobile_application_new.html', form=form, mobile_application=mobile_application)

@admin_blueprint.route('/mobile_application/latest', methods=['GET'])
@login_required
def mobile_application_latest():
    if current_user.role != User.ADMIN:
        return redirect(url_for('user.index'))

    mobile_application = MobileApplication.query.order_by(MobileApplication.id.desc()).first()
    return render_template('mobile_application_detail.html', mobile_application=mobile_application)

@admin_blueprint.route('/psysuite_application/new', methods=['GET', 'POST'])
@login_required
def psysuite_application_new():
    if current_user.role != User.ADMIN:
        return redirect(url_for('user.index'))

    form = MobileApplicationForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                mobile_application = PsysuiteApplication()
                mobile_application.version = form.version.data
                mobile_application.sver = form.sver.data
                mobile_application.description = form.description.data
                filename = 'psysuite%d.apk' % mobile_application.version
                filepath = os.path.join(app.instance_path, 'psysuite_applications', filename)
                mobile_application.apk_path = filepath
                f = form.apk.data
                f.save(filepath)
                db.session.add(mobile_application)
                db.session.commit()
                flash('New Psysuite application added')
                return redirect(url_for('admin.psysuite_application_latest'))
            except IntegrityError:
                db.session.rollback()
                flash('New Psysuite application added', 'error')
    mobile_application = PsysuiteApplication.query.order_by(PsysuiteApplication.id.desc()).first()
    if mobile_application:
        form.version.data = mobile_application.version + 1
        form.sver.data = mobile_application.sver
    return render_template('psysuite_application_new.html', form=form, mobile_application=mobile_application)

@admin_blueprint.route('/psysuite_application/latest', methods=['GET'])
@login_required
def psysuite_application_latest():
    if current_user.role != User.ADMIN:
        return redirect(url_for('user.index'))

    mobile_application = PsysuiteApplication.query.order_by(PsysuiteApplication.id.desc()).first()
    return render_template('psysuite_application_detail.html', mobile_application=mobile_application)


@admin_blueprint.route('/admin')
@login_required
def admin():
    if current_user.role != User.ADMIN:
        return redirect(url_for('user.index'))

    return render_template('admin.html')

@admin_blueprint.route('/stableupdate.xml')
def stableupdate():
    mobile_application = MobileApplication.query.order_by(MobileApplication.id.desc()).first()
    if not mobile_application:
        abort(404)

    stableupdate = mobile_application.stableupdate(request.host_url)
    
    return Response(response=stableupdate, status=200, mimetype="application/xml")

@admin_blueprint.route('/allspeak.apk')
def allspeak_apk():
    mobile_application = MobileApplication.query.order_by(MobileApplication.id.desc()).first()
    if not mobile_application:
        abort(404)

    return send_file(mobile_application.apk_path, mimetype='application/vnd.android.package-archive')


@admin_blueprint.route('/psysuitestableupdate.xml')
def psysuitestableupdate():
    print("update request")
    mobile_application = PsysuiteApplication.query.order_by(PsysuiteApplication.id.desc()).first()
    if not mobile_application:
        abort(404)

    stableupdate = mobile_application.stableupdate(request.host_url)

    return Response(response=stableupdate, status=200, mimetype="application/xml")

@admin_blueprint.route('/psysuite.apk')
def psysuite_apk():
    mobile_application = PsysuiteApplication.query.order_by(PsysuiteApplication.id.desc()).first()
    if not mobile_application:
        abort(404)

    return send_file(mobile_application.apk_path, mimetype='application/vnd.android.package-archive')