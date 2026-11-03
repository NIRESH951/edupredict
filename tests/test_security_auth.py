"""Module 1: password hashing, authentication, lock-out and account rules."""
import pytest

from core import database as db
from core.security import hash_password, needs_rehash, password_problems, verify_password
from services import auth_service
from services.auth_service import UserError


def test_hash_is_salted_pbkdf2_and_verifies():
    h1, h2 = hash_password("Secret@123"), hash_password("Secret@123")
    assert h1.startswith("pbkdf2_sha256$") and h1 != h2  # random salt
    assert verify_password("Secret@123", h1)
    assert not verify_password("secret@123", h1)
    assert not verify_password("Secret@123", "garbage")


def test_no_plain_text_passwords_in_database():
    rows = db.query_all("SELECT password_hash FROM users")
    assert rows and all(r["password_hash"].startswith("pbkdf2_sha256$") for r in rows)
    assert all("Admin@123" not in r["password_hash"] for r in rows)


def test_needs_rehash_for_weaker_hash():
    assert needs_rehash(hash_password("Secret@123", iterations=1000))
    assert not needs_rehash(hash_password("Secret@123"))


def test_password_policy():
    assert password_problems("Strong@123") == []
    assert len(password_problems("weak")) >= 3


@pytest.mark.parametrize("username,password,role", [
    ("admin", "Admin@123", "admin"), ("teacher1", "Teacher@123", "teacher"), ("student1", "Student@123", "student"),
])
def test_demo_accounts_log_in(username, password, role):
    user, msg = auth_service.authenticate(username, password)
    assert user is not None, msg
    assert user["role"] == role and "password_hash" not in user


def test_wrong_password_and_unknown_user_give_same_message():
    _, m1 = auth_service.authenticate("admin", "nope")
    _, m2 = auth_service.authenticate("nobody", "nope")
    assert m1 == m2 == "Invalid username or password."


def test_lockout_after_repeated_failures(admin_user):
    uid = auth_service.create_user("locktest", "Lock@1234", "admin", "Lock Test", created_by=admin_user)
    for _ in range(5):
        auth_service.authenticate("locktest", "bad")
    user, msg = auth_service.authenticate("locktest", "Lock@1234")
    assert user is None and "locked" in msg.lower()
    auth_service.unlock_user(uid, admin_user)
    user, _ = auth_service.authenticate("locktest", "Lock@1234")
    assert user is not None


def test_inactive_user_cannot_log_in(admin_user):
    uid = auth_service.create_user("inactive1", "Inactive@1", "admin", "Inactive User", created_by=admin_user)
    auth_service.set_active(uid, False, admin_user)
    user, msg = auth_service.authenticate("inactive1", "Inactive@1")
    assert user is None and "inactive" in msg.lower()


def test_change_password(admin_user):
    uid = auth_service.create_user("pwchange", "First@123", "admin", "Pw Change", created_by=admin_user)
    with pytest.raises(UserError):
        auth_service.change_password(uid, "wrong", "Second@123")
    with pytest.raises(UserError):
        auth_service.change_password(uid, "First@123", "weak")
    auth_service.change_password(uid, "First@123", "Second@123")
    assert auth_service.authenticate("pwchange", "Second@123")[0] is not None


def test_cannot_deactivate_or_delete_self(admin_user):
    with pytest.raises(UserError):
        auth_service.set_active(admin_user["id"], False, admin_user)
    with pytest.raises(UserError):
        auth_service.delete_user(admin_user["id"], admin_user)


def test_duplicate_username_and_role_links(admin_user):
    with pytest.raises(UserError):
        auth_service.create_user("admin", "Admin@1234", "admin", "Dup", created_by=admin_user)
    with pytest.raises(UserError):  # student account must link to a student
        auth_service.create_user("nolink", "Student@123", "student", "No Link", created_by=admin_user)
