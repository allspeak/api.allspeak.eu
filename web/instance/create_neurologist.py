# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import sys
import os

email = sys.argv[1]
password = sys.argv[2]

print('Creating neurologist %s...' % email)

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))


# Create the database tables, add some initial data, and commit to the database
from project import db
from project.models import User

# Insert user data
neurologist = User(email=email, plaintext_password=password, role=User.NEUROLOGIST)
db.session.add(neurologist)

# Commit the changes for the users
db.session.commit()

print('...done!')
