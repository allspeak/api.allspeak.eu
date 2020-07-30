# Check the PYTHONPATH environment variable before beginning to ensure that the
# top-level directory is included.  If not, append the top-level.  This allows
# the modules within the .../project/ directory to be discovered.
import os
import sys
import uuid
from shutil import copyfile
from shutil import rmtree
from flask_migrate import Config, command

if os.path.abspath(os.curdir) not in sys.path:
    sys.path.append(os.path.abspath(os.curdir))

from project import db, app, migrate
from project.models import User
from project.models import TrainingSession

# ----------------------------------------------------------------------------------------------------------------
# INIT DATA
# ----------------------------------------------------------------------------------------------------------------
admin_email = sys.argv[1]
admin_password = sys.argv[2]

neuro_email1 = sys.argv[3]
neuro_password1 = sys.argv[4]

neuro_email2 = sys.argv[5]
neuro_password2 = sys.argv[6]

neuro_email3 = sys.argv[7]
neuro_password3 = sys.argv[8]


n_patients = 1

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
psysuite_app_root = os.path.join(instance_path, 'psysuite_applications')
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

open(os.path.join(users_root, '.gitkeep'), 'a').close()
open(os.path.join(mobile_app_root, '.gitkeep'), 'a').close()
open(os.path.join(temp_train_data_root, '.gitkeep'), 'a').close()



print('Filesystem created')

# ----------------------------------------------------------------------------------------------------------------
# Create the database tables, add some initial data, and commit to the database
# ----------------------------------------------------------------------------------------------------------------

config = Config("migrations/alembic.ini")
config.set_main_option("script_location", "migrations")
with app.app_context():
    command.downgrade(config, 'base')
    command.upgrade(config, "head")

print('DB created')

# ----------------------------------------------------------------------------------------------------------------
# ADMIN (it also creates a folder to host the COMMON net   /.../users_data/ADMINK/train_data)
# ----------------------------------------------------------------------------------------------------------------
admin = User(role=User.ADMIN, email=admin_email, plaintext_password=admin_password)
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
training_session.net_path = os.path.join(dest_commonnet_path, 'net_273_252_280.pb')
training_session.completed = True

db.session.add(training_session)
db.session.commit()

copyfile(os.path.join(inputnet_path, 'net_273_252_280.pb'), training_session.net_path)
copyfile(os.path.join(inputnet_path, 'net_273_252_280.json'), os.path.join(dest_commonnet_path, 'net_273_252_280.json'))
copyfile(os.path.join(inputnet_path, 'vocabulary.json'), os.path.join(dest_commonnet_path, 'vocabulary.json'))

print('common training sessions created')

# ----------------------------------------------------------------------------------------------------------------
# NEUROLOGISTS
# ----------------------------------------------------------------------------------------------------------------
neuro1 = User(role=User.NEUROLOGIST, email=neuro_email1, plaintext_password=neuro_password1)
db.session.add(neuro1)
db.session.commit()
print('neurologist user created with api_key: ' + neuro1.api_key)

neuro2 = User(role=User.NEUROLOGIST, email=neuro_email2, plaintext_password=neuro_password2)
db.session.add(neuro2)
db.session.commit()
print('neurologist user created with api_key: ' + neuro2.api_key)

neuro3 = User(role=User.NEUROLOGIST, email=neuro_email3, plaintext_password=neuro_password3)
db.session.add(neuro3)
db.session.commit()
print('neurologist user created with api_key: ' + neuro3.api_key)

# ----------------------------------------------------------------------------------------------------------------
# PATIENTS
# ----------------------------------------------------------------------------------------------------------------
for i in range(0, n_patients):
    patient = User(role=User.PATIENT)
    db.session.add(patient)

db.session.commit()

print('patients created')


# ----------------------------------------------------------------------------------------------------------------
# END
# ----------------------------------------------------------------------------------------------------------------
