from flask_wtf import Form
from wtforms import StringField, IntegerField, TextAreaField
from wtforms.validators import InputRequired, Length, Optional
from flask_wtf.file import FileField, FileRequired
from project.models import User

class MobileApplicationForm(Form):
    version = IntegerField('Version Code', validators=[InputRequired()])
    sver = StringField('Version String', validators=[InputRequired(), Length(min=1, max=40)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)], render_kw={"rows": 7, "cols": 30})
    apk = FileField('APK', validators=[FileRequired()])