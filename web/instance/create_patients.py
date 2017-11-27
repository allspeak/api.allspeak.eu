# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import sys
import os

print('Creating patients...')

if len(sys.argv) == 1:
    print ('Usage: create_patients N')
    exit()

n = int(sys.argv[1])

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))


# Create the database tables, add some initial data, and commit to the database
from project import db
from project.models import User

for i in range(0, n):
    patient = User(role=User.PATIENT)

    db.session.add(patient)

db.session.commit()

print('...done!')
