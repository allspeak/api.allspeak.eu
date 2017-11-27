from project import db, bcrypt, app, images
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from datetime import datetime
from markdown import markdown
from flask import url_for
import bleach
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

class TrainingSession(db.Model):

    __tablename__ = "training_session"

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer, default=None, nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

    def __init__(self, user_id=None, device_id=None):
        self.user_id = user_id
        self.device_id = device_id

    def import_data(self, request):
        try:
            json_data = request.get_json()
            self.user_id = json_data['user_id']
            self.device_id = json_data['device_id']
        except KeyError as e:
            raise ValidationError('Invalid training session: missing ' + e.args[0])
        return self    
    
    def export_data(self):
        return {
            'self_url': self.get_url(),
            'id': self.id,
            'user_id': self.user_id,
            'device_id': self.device_id
        }

    def get_url(self):
        return url_for('training_api.get_training_session', recipe_id=self.id, _external=True)



class User(db.Model):

    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String, default=None, nullable=True)
    _password = db.Column(db.Binary(60), nullable=False)
    authenticated = db.Column(db.Boolean, default=False)
    registered_on = db.Column(db.DateTime, nullable=True)
    last_logged_in = db.Column(db.DateTime, nullable=True)
    current_logged_in = db.Column(db.DateTime, nullable=True)
    role = db.Column(db.String, default='user')
    training_sessions = db.relationship('TrainingSession', backref='user', lazy='dynamic')

    def __init__(self, email, plaintext_password, role='patient'):
        self.email = email
        self.password = plaintext_password
        self.authenticated = False
        self.registered_on = datetime.now()
        self.last_logged_in = None
        self.current_logged_in = datetime.now()
        self.role = role

    @hybrid_property
    def password(self):
        return self._password

    @password.setter
    def set_password(self, plaintext_password):
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

    def get_username(self):
        if self.role == 'patient':
            return 'HSR_' + self.get_id().zfill(3)
        else:
            return self.email

    def get_homepage(self):
        if self.role == 'admin':
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

