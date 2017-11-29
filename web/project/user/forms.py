from flask_wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length, EqualTo, Email
from project.models import User

class LoginForm(Form):
    email = StringField('Email', validators=[DataRequired(), Email(), Length(min=6, max=40)])
    password = PasswordField('Password', validators=[DataRequired()])


class EmailForm(Form):
    email = StringField('Email', validators=[DataRequired(), Email(), Length(min=6, max=40)])


class PasswordForm(Form):
    password = PasswordField('Password', validators=[DataRequired()])

class NewPatientForm(Form):
    pass
