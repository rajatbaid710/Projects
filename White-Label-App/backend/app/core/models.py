"""Model registry.

Alembic's autogenerate only sees tables whose model classes have been imported.
Importing them here — and importing *this* module from `alembic/env.py` — means
there is exactly one list to keep current, instead of a scatter of imports.

Add every new model module to this file as it is created.
"""

from app.core.db import Base

# No domain models yet — Phase 0 establishes the migration chain only.
# As modules land, import them here, e.g.:
#     from app.modules.users.models import User, Profile
# Such imports are unused by name, so they need a ruff/flake8 F401 exemption.

__all__ = ["Base"]
