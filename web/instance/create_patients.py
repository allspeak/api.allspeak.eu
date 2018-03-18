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

    # create user file system
    userkey = patient.get_key()
    user_path = os.path.join(os.curdir, 'instance', 'patients_data', userkey)

    print(user_path)

    wav_path = os.path.join(user_path, 'voicebank')
    train_path = os.path.join(user_path, 'train_data')
    recordings_path = os.path.join(user_path, 'recordings')
    os.makedirs(wav_path)
    os.makedirs(train_path)
    os.makedirs(recordings_path)    

db.session.commit()

print('...done!')
