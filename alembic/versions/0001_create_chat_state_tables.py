"""create chat state tables

Revision ID: 0001_create_chat_state_tables
Revises:
Create Date: 2026-05-02
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_create_chat_state_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "threads",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_threads_user_id", "threads", ["user_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("thread_id", sa.String(length=64), sa.ForeignKey("threads.id"), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_messages_thread_id", "messages", ["thread_id"])

    op.create_table(
        "thread_summaries",
        sa.Column("thread_id", sa.String(length=64), sa.ForeignKey("threads.id"), primary_key=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "user_memory",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("memory_key", sa.String(length=128), nullable=False),
        sa.Column("memory_value", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_user_memory_user_id", "user_memory", ["user_id"])
    op.create_unique_constraint("uq_user_memory_user_key", "user_memory", ["user_id", "memory_key"])


def downgrade() -> None:
    op.drop_constraint("uq_user_memory_user_key", "user_memory", type_="unique")
    op.drop_index("idx_user_memory_user_id", table_name="user_memory")
    op.drop_table("user_memory")
    op.drop_table("thread_summaries")
    op.drop_index("idx_messages_thread_id", table_name="messages")
    op.drop_table("messages")
    op.drop_index("idx_threads_user_id", table_name="threads")
    op.drop_table("threads")
    op.drop_table("users")
