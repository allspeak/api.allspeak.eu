from flask import render_template, Blueprint, request, redirect, url_for, flash, abort
from sqlalchemy.exc import IntegrityError
from flask_login import login_user, current_user, login_required, logout_user
from threading import Thread
from itsdangerous import URLSafeTimedSerializer
from datetime import datetime

from .forms import LoginForm, EmailForm, PasswordForm
from project import db, app
from project.models import User

user_blueprint = Blueprint('user', __name__)

def flash_errors(form):
    for field, errors in form.errors.items():
        for error in errors:
            flash(u"Error in the %s field - %s" % (
                getattr(form, field).label.text,
                error
            ), 'info')


################
#### routes ####
################


@user_blueprint.route('/', methods=['GET'])
@login_required
def index():
    return redirect(current_user.get_homepage())


@user_blueprint.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST':
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user is not None and user.is_correct_password(form.password.data):
                user.authenticated = True
                user.last_logged_in = user.current_logged_in
                user.current_logged_in = datetime.now()
                db.session.add(user)
                db.session.commit()
                login_user(user)
                flash('Thanks for logging in, {}'.format(current_user.email))
                if user.role == 'admin':
                    redirect_url = url_for('user.view_users')
                else:
                    redirect_url = url_for('user.view_patients')
                return redirect(redirect_url)
            else:
                flash('ERROR! Incorrect login credentials.', 'error')
    return render_template('login.html', form=form)


@user_blueprint.route('/logout')
@login_required
def logout():
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    flash('Goodbye!', 'info')
    return redirect(url_for('user.login'))


@user_blueprint.route('/user_profile/<id>')
@login_required
def user_profile(id):
    user = User.query.filter(User.id == id).first()
    return render_template('user_profile.html', user=user)


@user_blueprint.route('/email_change', methods=["GET", "POST"])
@login_required
def user_email_change():
    form = EmailForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                user_check = User.query.filter_by(email=form.email.data).first()
                if user_check is None:
                    user = current_user
                    user.email = form.email.data
                    db.session.add(user)
                    db.session.commit()
                    return redirect(url_for('user.user_profile', id=user.id))
                else:
                    flash('Sorry, that email already exists!', 'error')
            except IntegrityError:
                flash('Error! That email already exists!', 'error')
    return render_template('email_change.html', form=form)


@user_blueprint.route('/password_change', methods=["GET", "POST"])
@login_required
def user_password_change():
    form = PasswordForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            user = current_user
            user.password = form.password.data
            db.session.add(user)
            db.session.commit()
            flash('Password has been updated!', 'success')
            return redirect(url_for('user.user_profile', id=user.id))

    return render_template('password_change.html', form=form)


@user_blueprint.route('/view_patients')
@login_required
def view_patients():
    if current_user.role == 'patient':
        abort(403)
    else:
        users = User.query.filter(User.role == 'patient').order_by(User.id).all()
        return render_template('view_patients.html', users=users)


@user_blueprint.route('/view_users')
@login_required
def view_users():
    if current_user.role != 'admin':
        abort(403)
    else:
        roles_to_display =  ['patient', 'neurologist', 'admin']     
        users = User.query.filter(User.role.in_(roles_to_display)).order_by(User.id).all()
        return render_template('view_users.html', users=users)