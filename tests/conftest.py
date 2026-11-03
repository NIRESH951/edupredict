"""
Test configuration: every test session uses a fresh temporary database and
model folder, seeded once with the synthetic demo data and trained models.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="edupredict_test_"))
os.environ["EDUPREDICT_DB"] = str(_TMP / "test.db")
os.environ["EDUPREDICT_MODEL_DIR"] = str(_TMP / "models")
os.environ["EDUPREDICT_DATA_DIR"] = str(_TMP / "data")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    from core.seed import seed_database

    summary = seed_database()
    assert summary["seeded"]
    return summary


@pytest.fixture
def admin_user():
    from services.auth_service import get_user_by_username, public_user

    return public_user(get_user_by_username("admin"))


@pytest.fixture
def admin_scope():
    from core.scope import Scope

    return Scope.admin()


@pytest.fixture
def teacher_scope():
    from core.scope import Scope
    from services.auth_service import get_user_by_username

    return Scope.from_user(get_user_by_username("teacher1"))


@pytest.fixture
def student_scope():
    from core.scope import Scope
    from services.auth_service import get_user_by_username

    return Scope.from_user(get_user_by_username("student1"))
