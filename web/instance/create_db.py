# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import sys
import os

print('Creating database tables for AllSpeak...')

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))


# Create the database tables, add some initial data, and commit to the database
from project import db
from project.models import User


# Drop all of the existing database tables
db.drop_all()

# Create the database and the database table
db.create_all()

# Commit the changes for the users
db.session.commit()

print('...done!')
