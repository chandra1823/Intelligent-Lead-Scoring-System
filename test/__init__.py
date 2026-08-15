"""
Test package setup.

These environment variables must be set before anything imports
app.core.config, because the database engine is built from settings at import
time. Putting them here guarantees it: unittest imports the package before any
module inside it, whatever order the individual test modules load in.
"""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_TMPDIR = tempfile.mkdtemp(prefix="leadscoring-tests-")

os.environ.setdefault(
    "LEAD_API_DATABASE_URL", f"sqlite:///{Path(_TMPDIR) / 'leadscoring-tests.db'}"
)
os.environ.setdefault("LEAD_API_ARTIFACT_ROOT", str(ROOT / "artifacts"))
os.environ.setdefault("LEAD_API_RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LEAD_API_SECRET_KEY", "test-only-key")
