"""initial_schema

Revision ID: 8ee4d82ef1a1
Revises: None
Create Date: 2018-08-17 12:51:06.531683

"""

# revision identifiers, used by Alembic.
revision = '8ee4d82ef1a1'
down_revision = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.create_table('mobile_application',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=False),
    sa.Column('sver', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('apk_path', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('version')
    )
    op.create_table('user',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('email', sa.String(), nullable=True),
    sa.Column('_password', sa.Binary(), nullable=True),
    sa.Column('authenticated', sa.Boolean(), nullable=True),
    sa.Column('registered_on', sa.DateTime(), nullable=True),
    sa.Column('last_logged_in', sa.DateTime(), nullable=True),
    sa.Column('current_logged_in', sa.DateTime(), nullable=True),
    sa.Column('role', sa.String(), nullable=False),
    sa.Column('api_key', sa.String(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('api_key'),
    sa.UniqueConstraint('email')
    )
    op.create_table('device',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('uuid', sa.String(), nullable=True),
    sa.Column('model', sa.String(), nullable=True),
    sa.Column('manufacturer', sa.String(), nullable=True),
    sa.Column('serial', sa.String(), nullable=True),
    sa.Column('version', sa.String(), nullable=True),
    sa.Column('platform', sa.String(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('registered_on', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('training_session',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_uid', sa.String(), nullable=False),
    sa.Column('model_type', sa.Integer(), nullable=False),
    sa.Column('preproc_type', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('completed', sa.Boolean(), nullable=False),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.Column('net_path', sa.String(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_uid')
    )
    op.create_table('error',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_uid', sa.String(), nullable=True),
    sa.Column('error_code', sa.Integer(), nullable=False),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('created_on', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_uid')
    )
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('error')
    op.drop_table('training_session')
    op.drop_table('device')
    op.drop_table('user')
    op.drop_table('mobile_application')
    ### end Alembic commands ###