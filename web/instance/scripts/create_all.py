# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import os
import sys
import uuid
from shutil import copyfile
from shutil import rmtree

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))

from project import db
from project.models import User
from project.models import TrainingSession
# ----------------------------------------------------------------------------------------------------------------
# INIT DATA
# ----------------------------------------------------------------------------------------------------------------
admin_email = 'alberto.inuggi@gmail.com'
admin_password = '1234'

admin_key = 'ADMINK'
neuro_email = 'alberto.inuggi@iit.it'
neuro_password = '1234'

n_patients = 5

common_net_type = 273
preproc_type = 252

recreate_filesystem = True
# ----------------------------------------------------------------------------------------------------------------
# CREATE FILE SYSTEM
# ----------------------------------------------------------------------------------------------------------------
web_path = os.curdir
inputnet_path = os.path.join(web_path, 'inputnet')
instance_path = os.path.join(os.curdir, 'instance')

users_root = os.path.join(instance_path, 'users_data')
mobile_app_root = os.path.join(instance_path, 'mobile_applications')
temp_train_data_root = os.path.join(instance_path, 'temp_train_data')

if recreate_filesystem is True:
    rmtree(users_root)
    rmtree(mobile_app_root)
    rmtree(temp_train_data_root)

if os.path.exists(users_root) is False:
    os.makedirs(users_root)

if os.path.exists(mobile_app_root) is False:
    os.makedirs(mobile_app_root)

if os.path.exists(temp_train_data_root) is False:
    os.makedirs(temp_train_data_root)

print('Filesystem created')

# ----------------------------------------------------------------------------------------------------------------
# Create the database tables, add some initial data, and commit to the database
# ----------------------------------------------------------------------------------------------------------------
db.drop_all()       # Drop all of the existing database tables
db.create_all()     # Create the database and the database table
db.session.commit() # Commit the changes for the users

print('DB created')

# ----------------------------------------------------------------------------------------------------------------
# ADMIN (it also creates a folder to host the COMMON net   /.../users_data/ADMINK/train_data)
# ----------------------------------------------------------------------------------------------------------------
admin = User(role=User.ADMIN, email=admin_email, plaintext_password=admin_password, apikey=admin_key)
db.session.add(admin)
db.session.commit()

print('admin user created with api_key: ' + admin.api_key)
# ----------------------------------------------------------------------------------------------------------------
# ADD COMMON NET ASSOCIATED TO ADMIN
# ----------------------------------------------------------------------------------------------------------------
session_uid = uuid.uuid1()
admin_id = admin.id
dest_commonnet_path = os.path.join(admin.get_userpath(), 'train_data', str(session_uid))
os.makedirs(dest_commonnet_path)

training_session = TrainingSession(session_uid, common_net_type, preproc_type, admin_id)
training_session.net_path = os.path.join(dest_commonnet_path, 'controls_fsc.pb')

db.session.add(training_session)
db.session.commit()

copyfile(os.path.join(inputnet_path, 'controls_fsc.pb'), training_session.net_path)
copyfile(os.path.join(inputnet_path, 'vocabulary.json'), os.path.join(dest_commonnet_path, 'vocabulary.json'))

print('common training sessions created')

# ----------------------------------------------------------------------------------------------------------------
# NEUROLOGIST
# ----------------------------------------------------------------------------------------------------------------
neuro = User(role=User.NEUROLOGIST, email=neuro_email, plaintext_password=neuro_password)
db.session.add(neuro)
db.session.commit()

print('neurologist user created with api_key: ' + neuro.api_key)
# ----------------------------------------------------------------------------------------------------------------
# PATIENTS
# ----------------------------------------------------------------------------------------------------------------
for i in range(0, n_patients):
    patient = User(role=User.PATIENT)
    db.session.add(patient)

db.session.commit()

print('patients created')