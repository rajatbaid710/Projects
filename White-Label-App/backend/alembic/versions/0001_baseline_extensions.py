"""baseline: required postgres extensions

Establishes the migration chain and installs the extensions the schema will
depend on from Phase 1 onward:

* ``citext``   — case-insensitive text. Emails are the login identifier, and
  ``Bob@x.com`` and ``bob@x.com`` must not be two accounts. Enforcing that with
  a citext column plus a unique index is reliable; remembering to ``.lower()``
  at every call site is not.
* ``pgcrypto`` — ``gen_random_uuid()`` for server-side UUID primary keys.

Deliberately no tables: the multi-tenancy decision (a ``tenant_id`` on every
row vs. one deployment per client) is still open, and it determines the shape of
every table that follows. Making it before the first real migration is much
cheaper than a data migration afterwards.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS citext")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS pgcrypto")
    op.execute("DROP EXTENSION IF EXISTS citext")
