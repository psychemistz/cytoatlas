"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2026-01-28

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Atlases table
    op.create_table(
        'atlases',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False, server_default='1.0.0'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('n_cells', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_samples', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_cell_types', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('h5ad_path', sa.Text(), nullable=True),
        sa.Column('results_path', sa.Text(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index('ix_atlases_name', 'atlases', ['name'])
    op.create_index('ix_atlases_status', 'atlases', ['status'])

    # Users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('hashed_password', sa.String(255), nullable=False),
        sa.Column('api_key_hash', sa.String(255), nullable=True),
        sa.Column('full_name', sa.String(255), nullable=True),
        sa.Column('institution', sa.String(255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('is_admin', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('quota_requests_per_day', sa.Integer(), nullable=False, server_default='10000'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )
    op.create_index('ix_users_email', 'users', ['email'])

    # Signatures table
    op.create_table(
        'signatures',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('signature_type', sa.String(50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('n_genes', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('gene_list', sa.Text(), nullable=True),
        sa.Column('category', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_signatures_name', 'signatures', ['name'])
    op.create_index('ix_signatures_signature_type', 'signatures', ['signature_type'])
    op.create_index('ix_signatures_category', 'signatures', ['category'])

    # Samples table
    op.create_table(
        'samples',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('atlas_id', sa.Integer(), nullable=False),
        sa.Column('sample_id', sa.String(100), nullable=False),
        sa.Column('donor_id', sa.String(100), nullable=True),
        sa.Column('sex', sa.String(20), nullable=True),
        sa.Column('age', sa.Float(), nullable=True),
        sa.Column('bmi', sa.Float(), nullable=True),
        sa.Column('disease', sa.String(200), nullable=True),
        sa.Column('disease_group', sa.String(100), nullable=True),
        sa.Column('condition', sa.String(100), nullable=True),
        sa.Column('therapy', sa.String(200), nullable=True),
        sa.Column('therapy_response', sa.String(50), nullable=True),
        sa.Column('timepoint', sa.String(50), nullable=True),
        sa.Column('tissue', sa.String(100), nullable=True),
        sa.Column('organ', sa.String(100), nullable=True),
        sa.Column('n_cells', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cohort', sa.String(50), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['atlas_id'], ['atlases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_samples_atlas_id', 'samples', ['atlas_id'])
    op.create_index('ix_samples_sample_id', 'samples', ['sample_id'])
    op.create_index('ix_samples_donor_id', 'samples', ['donor_id'])
    op.create_index('ix_samples_disease', 'samples', ['disease'])
    op.create_index('ix_samples_disease_group', 'samples', ['disease_group'])
    op.create_index('ix_samples_therapy_response', 'samples', ['therapy_response'])
    op.create_index('ix_samples_tissue', 'samples', ['tissue'])
    op.create_index('ix_samples_organ', 'samples', ['organ'])
    op.create_index('ix_samples_cohort', 'samples', ['cohort'])
    op.create_index('ix_samples_sex', 'samples', ['sex'])

    # Cell types table
    op.create_table(
        'cell_types',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('atlas_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('harmonized_name', sa.String(200), nullable=True),
        sa.Column('lineage', sa.String(100), nullable=True),
        sa.Column('parent_type', sa.String(200), nullable=True),
        sa.Column('n_cells', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('n_samples', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('organ', sa.String(100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['atlas_id'], ['atlases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_cell_types_atlas_id', 'cell_types', ['atlas_id'])
    op.create_index('ix_cell_types_name', 'cell_types', ['name'])
    op.create_index('ix_cell_types_harmonized_name', 'cell_types', ['harmonized_name'])
    op.create_index('ix_cell_types_lineage', 'cell_types', ['lineage'])
    op.create_index('ix_cell_types_organ', 'cell_types', ['organ'])

    # Computed stats table
    op.create_table(
        'computed_stats',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('atlas_id', sa.Integer(), nullable=False),
        sa.Column('stat_type', sa.String(50), nullable=False),
        sa.Column('grouping_key1', sa.String(100), nullable=True),
        sa.Column('grouping_key2', sa.String(100), nullable=True),
        sa.Column('grouping_value1', sa.String(200), nullable=True),
        sa.Column('grouping_value2', sa.String(200), nullable=True),
        sa.Column('signature', sa.String(100), nullable=False),
        sa.Column('signature_type', sa.String(50), nullable=False),
        sa.Column('metric', sa.String(50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('value2', sa.Float(), nullable=True),
        sa.Column('value3', sa.Float(), nullable=True),
        sa.Column('n_samples', sa.Integer(), nullable=True),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['atlas_id'], ['atlases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_computed_stats_atlas_id', 'computed_stats', ['atlas_id'])
    op.create_index('ix_computed_stats_stat_type', 'computed_stats', ['stat_type'])
    op.create_index('ix_computed_stats_grouping_key1', 'computed_stats', ['grouping_key1'])
    op.create_index('ix_computed_stats_grouping_key2', 'computed_stats', ['grouping_key2'])
    op.create_index('ix_computed_stats_grouping_value1', 'computed_stats', ['grouping_value1'])
    op.create_index('ix_computed_stats_signature', 'computed_stats', ['signature'])
    op.create_index('ix_computed_stats_signature_type', 'computed_stats', ['signature_type'])
    op.create_index('ix_computed_stats_metric', 'computed_stats', ['metric'])

    # Validation metrics table
    op.create_table(
        'validation_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('atlas_id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(50), nullable=False),
        sa.Column('cell_type', sa.String(200), nullable=True),
        sa.Column('signature', sa.String(100), nullable=False),
        sa.Column('signature_type', sa.String(50), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('ci_lower', sa.Float(), nullable=True),
        sa.Column('ci_upper', sa.Float(), nullable=True),
        sa.Column('n_samples', sa.Integer(), nullable=True),
        sa.Column('genes_detected', sa.Integer(), nullable=True),
        sa.Column('genes_total', sa.Integer(), nullable=True),
        sa.Column('coverage_pct', sa.Float(), nullable=True),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['atlas_id'], ['atlases.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_validation_metrics_atlas_id', 'validation_metrics', ['atlas_id'])
    op.create_index('ix_validation_metrics_metric_type', 'validation_metrics', ['metric_type'])
    op.create_index('ix_validation_metrics_cell_type', 'validation_metrics', ['cell_type'])
    op.create_index('ix_validation_metrics_signature', 'validation_metrics', ['signature'])
    op.create_index('ix_validation_metrics_signature_type', 'validation_metrics', ['signature_type'])


def downgrade() -> None:
    op.drop_table('validation_metrics')
    op.drop_table('computed_stats')
    op.drop_table('cell_types')
    op.drop_table('samples')
    op.drop_table('signatures')
    op.drop_table('users')
    op.drop_table('atlases')
