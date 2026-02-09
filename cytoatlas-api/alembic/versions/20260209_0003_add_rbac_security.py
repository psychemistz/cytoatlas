"""Add RBAC and security enhancements

Revision ID: 0003
Revises: 0002
Create Date: 2026-02-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add role column to users table
    op.add_column('users', sa.Column('role', sa.String(50), nullable=False, server_default='viewer'))

    # Add API key prefix for O(1) lookup
    op.add_column('users', sa.Column('api_key_prefix', sa.String(8), nullable=True))
    op.create_index('ix_users_api_key_prefix', 'users', ['api_key_prefix'])

    # Add API key timestamps
    op.add_column('users', sa.Column('api_key_created_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('users', sa.Column('api_key_last_used', sa.DateTime(timezone=True), nullable=True))

    # Drop is_admin column (now computed from role)
    # Note: SQLite doesn't support DROP COLUMN, so we skip this for SQLite databases
    # For PostgreSQL, uncomment the following line:
    # op.drop_column('users', 'is_admin')


def downgrade() -> None:
    # Remove new columns
    op.drop_index('ix_users_api_key_prefix', 'users')
    op.drop_column('users', 'api_key_last_used')
    op.drop_column('users', 'api_key_created_at')
    op.drop_column('users', 'api_key_prefix')
    op.drop_column('users', 'role')

    # Re-add is_admin column if needed
    # op.add_column('users', sa.Column('is_admin', sa.Boolean(), nullable=False, server_default=sa.false()))
