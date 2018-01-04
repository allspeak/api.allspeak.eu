import sys, os

wd = os.path.dirname(__file__)

if os.path.abspath(wd) not in sys.path:
    sys.path.append(wd)

activate_this = wd + '/../env/bin/activate_this.py'
with open(activate_this) as file_:
    exec(file_.read(), dict(__file__=activate_this))

from project import app as application