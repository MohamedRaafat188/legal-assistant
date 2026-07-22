"""add summarized_through_turn to conversations

Revision ID: d13f9d6b4f9a
Revises: 52cf2fe53da2
Create Date: 2026-07-22 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd13f9d6b4f9a'
down_revision: Union[str, Sequence[str], None] = '52cf2fe53da2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'conversations',
        sa.Column('summarized_through_turn', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('conversations', 'summarized_through_turn')
