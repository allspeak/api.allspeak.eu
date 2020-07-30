from project import db, bcrypt, app, images
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from datetime import datetime
from markdown import markdown
from flask import url_for
import bleach
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
import random
import string
import os

format="%Y-%m-%d %H:%M:%S"

class TrainingSession(db.Model):

    __tablename__ = "training_session"

    id = db.Column(db.Integer, primary_key=True)
    session_uid = db.Column(db.String, nullable=False, unique=True)
    model_type = db.Column(db.Integer, nullable=False)
    preproc_type = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    completed = db.Column(db.Boolean, default=False, nullable=False)
    created_on = db.Column(db.DateTime, nullable=True)
    net_path = db.Column(db.String, nullable=False)

    def __init__(self, session_uid, model_type, preproc_type, user_id=None):
        self.user_id = user_id
        self.session_uid = session_uid
        self.model_type = model_type
        self.preproc_type = preproc_type
        self.completed = False
        self.created_on = datetime.now()  
        self.net_path = ""
    
    def export_data(self):
        return {
            'self_url': self.get_url(),
            'id': self.id,
            'user_id': self.user_id,
            'session_uid': self.session_uid,
            'model_type': self.model_type,
            'preproc_type': self.preproc_type,
            'completed': self.completed,
            'created_on': self.created_on,
            'net_path': self.net_path
        }

    def get_url(self):
        return url_for('training_api.get_training_session', session_uid=self.session_uid, _external=True)


class User(db.Model):

    __tablename__ = "user"
    PATIENT = 'patient'
    NEUROLOGIST = 'neurologist'
    ADMIN = 'admin'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, default=None, nullable=True, unique=True)
    _password = db.Column(db.Binary(60), nullable=True)
    authenticated = db.Column(db.Boolean, default=False)
    registered_on = db.Column(db.DateTime, nullable=True)
    last_logged_in = db.Column(db.DateTime, nullable=True)
    current_logged_in = db.Column(db.DateTime, nullable=True)
    role = db.Column(db.String, default=PATIENT, nullable=False)
    api_key = db.Column(db.String, nullable=True, unique=True)
    training_sessions = db.relationship('TrainingSession', backref='user', lazy='dynamic')

    def __init__(self, role, email = None, plaintext_password = None, apikey = None):

        self.email = email
        self.password = plaintext_password
        self.authenticated = False
        self.registered_on = datetime.now()
        self.last_logged_in = None
        self.current_logged_in = None
        self.role = role
        if apikey is None:      # used to set specific api_key to special users (like ADMIN)
            self.regenerate_api_key()
        else:
            self.api_key = apikey
        self.create_filesystem()

    # create user file system (NEUROLOGIST does not have their own folder)
    def create_filesystem(self):
        
        user_path = self.get_userpath()
        if self.role == User.PATIENT:
            wav_path = os.path.join(user_path, 'voicebank')
            train_path = os.path.join(user_path, 'train_data')
            recordings_path = os.path.join(user_path, 'recordings')
            os.makedirs(wav_path)
            os.makedirs(train_path)
            os.makedirs(recordings_path) 
        elif self.role == User.ADMIN:
            train_path = os.path.join(user_path, 'train_data')
            os.makedirs(train_path)

    def get_userpath(self):
        return os.path.join(app.instance_path, 'users_data', self.api_key)

    @hybrid_property
    def password(self):
        return self._password

    @password.setter
    def set_password(self, plaintext_password):
        if plaintext_password is None:
            self._password = None
        else:
            self._password = bcrypt.generate_password_hash(plaintext_password)

    @hybrid_method
    def is_correct_password(self, plaintext_password):
        return bcrypt.check_password_hash(self.password, plaintext_password)

    @property
    def is_authenticated(self):
        """Return True if the user is authenticated."""
        return self.authenticated

    @property
    def is_active(self):
        """Always True, as all users are active."""
        return True

    @property
    def is_anonymous(self):
        """Always False, as anonymous users aren't supported."""
        return False

    def get_id(self):
        """Return the email address to satisfy Flask-Login's requirements."""
        """Requires use of Python 3"""
        return str(self.id)

    def get_key(self):
        """Return the api_key."""
        return self.api_key

    def get_username(self):
        if self.role == User.PATIENT:
            return 'HSR_' + self.get_id().zfill(3)
        else:
            return self.email

    def get_homepage(self):
        if self.role == User.ADMIN:
            return url_for('user.view_users')
        else:
            return url_for('user.view_patients')
        
    def generate_auth_token(self, expires_in=3600):
        s = Serializer(app.config['SECRET_KEY'], expires_in=expires_in)
        return s.dumps({'id': self.id}).decode('utf-8')

    @staticmethod
    def verify_auth_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            data = s.loads(token)
        except:
            return None
        return User.query.get(data['id'])

    def __repr__(self):
        return '<User {}>'.format(self.email)

    def regenerate_api_key(self):

        if self.api_key is not None:
            current_path = self.get_userpath()  # NOT USER CREATION: get user current path before changing it
        else:
            current_path = None
        
        tempkey = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(app.config['NDIGITS_APIKEY']))
        user = User.query.filter(User.api_key == tempkey).first()
        if user is None:
            self.api_key = tempkey

            # renames user folder if exist (basically, on user creation it doesn't do anything)
            new_path = self.get_userpath()

            if os.path.exists(new_path) is True: 
                # raise Exception(msg)
                return False # "ERROR" # TODO: rise an exception....new_path should not be present

            if current_path is not None:
                if os.path.exists(current_path) is True: 
                    os.renames(current_path, new_path)
            return True       
        else:
            return self.regenerate_api_key()

    def refresh_login(self):
        self.last_logged_in = self.current_logged_in
        self.current_logged_in = datetime.now()


class Device(db.Model):

    __tablename__ = "device"

    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String, default=None, nullable=True)
    model = db.Column(db.String, default=None, nullable=True)
    manufacturer = db.Column(db.String, default=None, nullable=True)
    serial = db.Column(db.String, default=None, nullable=True)
    version = db.Column(db.String, default=None, nullable=True)
    platform = db.Column(db.String, default=None, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    registered_on = db.Column(db.DateTime, nullable=True)

    def __init__(self):
        self.registered_on = datetime.now()

    def import_data(self, user, request):
        try:
            json_data = request.get_json()
            self.uuid = json_data['uuid']
            self.model = json_data['model']
            self.manufacturer = json_data['manufacturer']
            self.serial = json_data['serial']
            self.version = json_data['version']
            self.platform = json_data['platform']
            self.user_id = user.id
        except KeyError as e:
            raise ValidationError('Invalid device: missing ' + e.args[0])
        return self

    def export_data(self):
        return {
            'self_url': self.get_url(),
            'uuid': self.uuid,
            'model': self.model,
            'manufacturer': self.manufacturer,
            'serial': self.serial,
            'version': self.version,
            'platform': self.platform,
            'user_id': self.user_id,
            'registered_on': self.get_registered_on_str()
        }

    def get_url(self):
        return url_for('user_api.get_device', uuid=self.uuid, _external=True)

    def get_registered_on_str(self):
        if self.registered_on is None:
            return None
        else:
            return self.registered_on.strftime(format)


class MobileApplication(db.Model):

    __tablename__ = "mobile_application"

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.Integer, default=None, nullable=False, unique=True)
    sver = db.Column(db.String, default=None, nullable=True)
    description = db.Column(db.String, default=None, nullable=True)
    apk_path = db.Column(db.String, default=None, nullable=True)

    def stableupdate(self, host_url):
        return '''
        <update>
            <version>%d</version>
            <sver>%s</sver>
            <description>%s</description>
            <name>AllSpeak</name>
            <url>%sallspeak.apk</url>
        </update>
        ''' % (self.version, self.sver, self.description, host_url)


class Error(db.Model):

    __tablename__ = "error"

    id = db.Column(db.Integer, primary_key=True)
    session_uid = db.Column(db.String, nullable=True, unique=True)
    error_code = db.Column(db.Integer, nullable=False)
    description = db.Column(db.String, default=None, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_on = db.Column(db.DateTime, nullable=True)

    def __init__(self, session_uid, error_code, description, user_id):
        self.session_uid = session_uid
        self.description = description
        self.error_code = error_code
        self.created_on = datetime.now()
        self.user_id = user_id
        


class PsysuiteApplication(db.Model):

    __tablename__ = "psysuite_mobile_application"

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.Integer, default=None, nullable=False, unique=True)
    sver = db.Column(db.String, default=None, nullable=True)
    description = db.Column(db.String, default=None, nullable=True)
    apk_path = db.Column(db.String, default=None, nullable=True)

    def stableupdate(self, host_url):
        return '''
        <update>
            <version>%d</version>
            <sver>%s</sver>
            <description>%s</description>
            <name>PsySuite</name>
            <url>%spsysuite.apk</url>
        </update>
        ''' % (self.version, self.sver, self.description, host_url)


    