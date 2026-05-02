"""add auth columns to users

Revision ID: 0002_add_auth_columns_to_users
Revises: 0001_create_chat_state_tables
Create Date: 2026-05-02
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0002_add_auth_columns_to_users"
down_revision = "0001_create_chat_state_tables"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("password_hash", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "is_active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("true"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.func.now(),
            )
        )
        batch_op.create_unique_constraint("uq_users_email", ["email"])

    # Drop server defaults after backfill to keep app-layer defaults authoritative.
    with op.batch_alter_table("users") as batch_op:
        batch_op.alter_column("is_active", server_default=None)
        batch_op.alter_column("updated_at", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("uq_users_email", type_="unique")
        batch_op.drop_column("updated_at")
        batch_op.drop_column("is_active")
        batch_op.drop_column("password_hash")
