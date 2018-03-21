# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import sys
import os

if len(sys.argv) < 3:
    print ('Usage: create_admin email password')
    exit()

email = sys.argv[1]
password = sys.argv[2]

print('Creating admin %s...' % email)

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))


# Create the database tables, add some initial data, and commit to the database
from project import db
from project.models import User

# Insert user data
admin = User(role=User.ADMIN, email=email, plaintext_password=password)
db.session.add(admin)

# Commit the changes for the users
db.session.commit()

print('...done!')
