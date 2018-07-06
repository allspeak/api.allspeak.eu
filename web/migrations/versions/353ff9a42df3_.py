"""empty message

Revision ID: 353ff9a42df3
Revises: 813cf34bfa87
Create Date: 2018-06-05 10:18:57.841685

"""

# revision identifiers, used by Alembic.
revision = '353ff9a42df3'
down_revision = '813cf34bfa87'

from alembic import op
import sqlalchemy as sa


def upgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('error', 'session_uid',
               existing_type=sa.VARCHAR(),
               nullable=True)
    ### end Alembic commands ###


def downgrade():
    ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('error', 'session_uid',
               existing_type=sa.VARCHAR(),
               nullable=False)
    ### end Alembic commands ###