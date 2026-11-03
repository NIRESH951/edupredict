"""
Authentication and user-account management (Module 1).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

import config
from core import database as db
from core.security import hash_password, needs_rehash, password_problems, verify_password
from core.validation import valid_email, valid_username

_TIME_FMT = "%Y-%m-%d %H:%M:%S"
_GENERIC_LOGIN_ERROR = "Invalid username or password."
_DUMMY_HASH: str | None = None


class UserError(ValueError):
    """Raised for invalid user-management operations (message is user-facing)."""


def _now() -> str:
    return datetime.now().strftime(_TIME_FMT)


def public_user(row: dict) -> dict:
    """Strip secrets before a user dict is placed in the session."""
    return {k: v for k, v in row.items() if k not in ("password_hash", "failed_attempts", "locked_until")}


def get_user(user_id: int) -> dict | None:
    return db.query_one("SELECT * FROM users WHERE id = ?", [user_id])


def get_user_by_username(username: str) -> dict | None:
    return db.query_one("SELECT * FROM users WHERE username = ?", [username.strip()])


def authenticate(username: str, password: str) -> tuple[dict | None, str]:
    """
    Verify credentials. Returns (user, message).
    Applies account lock-out after repeated failures and refuses inactive accounts.
    """
    username = (username or "").strip()
    if not username or not password:
        return None, "Please enter both username and password."

    user = get_user_by_username(username)
    if user is None:
        # Spend the same time as a real check so usernames cannot be probed by timing.
        global _DUMMY_HASH
        if _DUMMY_HASH is None:
            _DUMMY_HASH = hash_password("timing-equaliser")
        verify_password(password, _DUMMY_HASH)
        db.log_activity(None, username, "LOGIN_FAILED", "unknown username")
        return None, _GENERIC_LOGIN_ERROR

    if user["locked_until"]:
        locked_until = datetime.strptime(user["locked_until"], _TIME_FMT)
        if locked_until > datetime.now():
            minutes = max(1, int((locked_until - datetime.now()).total_seconds() // 60) + 1)
            return None, f"Account temporarily locked after repeated failed attempts. Try again in {minutes} minute(s)."

    if not verify_password(password, user["password_hash"]):
        attempts = int(user["failed_attempts"] or 0) + 1
        locked_until = None
        if attempts >= config.MAX_FAILED_LOGINS:
            locked_until = (datetime.now() + timedelta(minutes=config.LOCKOUT_MINUTES)).strftime(_TIME_FMT)
            attempts = 0
        db.execute(
            "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE id = ?",
            [attempts, locked_until, user["id"]],
        )
        db.log_activity(user["id"], username, "LOGIN_FAILED", "wrong password")
        if locked_until:
            return None, f"Too many failed attempts. Account locked for {config.LOCKOUT_MINUTES} minutes."
        return None, _GENERIC_LOGIN_ERROR

    if not user["is_active"]:
        db.log_activity(user["id"], username, "LOGIN_BLOCKED", "inactive account")
        return None, "This account is inactive. Please contact the administrator."

    updates = {"failed_attempts": 0, "locked_until": None, "last_login": _now()}
    if needs_rehash(user["password_hash"]):
        updates["password_hash"] = hash_password(password)
    sets = ", ".join(f"{k} = ?" for k in updates)
    db.execute(f"UPDATE users SET {sets} WHERE id = ?", [*updates.values(), user["id"]])
    db.log_activity(user["id"], username, "LOGIN", f"role={user['role']}")
    fresh = get_user(user["id"])
    return public_user(fresh), "Login successful."


def change_password(user_id: int, current_password: str, new_password: str) -> None:
    user = get_user(user_id)
    if user is None:
        raise UserError("User not found.")
    if not verify_password(current_password, user["password_hash"]):
        raise UserError("Current password is incorrect.")
    if current_password == new_password:
        raise UserError("New password must be different from the current password.")
    problems = password_problems(new_password)
    if problems:
        raise UserError("Password must contain " + ", ".join(problems) + ".")
    db.execute(
        "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
        [hash_password(new_password), user_id],
    )
    db.log_activity(user_id, user["username"], "PASSWORD_CHANGED")


def create_user(
    username: str,
    password: str,
    role: str,
    full_name: str,
    email: str | None = None,
    *,
    student_id: str | None = None,
    teacher_id: str | None = None,
    must_change_password: bool = False,
    is_active: bool = True,
    created_by: dict | None = None,
    enforce_policy: bool = True,
) -> int:
    username = (username or "").strip()
    if not valid_username(username):
        raise UserError("Username must be 3-30 characters: letters, digits, dot, underscore or hyphen.")
    if role not in config.ROLES:
        raise UserError("Invalid role.")
    if not full_name or len(full_name.strip()) < 3:
        raise UserError("Full name must have at least 3 characters.")
    if email and not valid_email(email):
        raise UserError("Email address is not valid.")
    if enforce_policy:
        problems = password_problems(password)
        if problems:
            raise UserError("Password must contain " + ", ".join(problems) + ".")
    if role == "student" and not student_id:
        raise UserError("A student account must be linked to a student profile.")
    if role == "teacher" and not teacher_id:
        raise UserError("A teacher account must be linked to a teacher profile.")
    if role == "student":
        existing = db.query_one("SELECT username FROM users WHERE student_id = ?", [student_id])
        if existing:
            raise UserError(f"This student already has a login account ('{existing['username']}').")
    if role == "teacher":
        existing = db.query_one("SELECT username FROM users WHERE teacher_id = ?", [teacher_id])
        if existing:
            raise UserError(f"This teacher already has a login account ('{existing['username']}').")
    try:
        user_id = db.execute(
            """INSERT INTO users (username, password_hash, role, full_name, email, is_active,
                                  student_id, teacher_id, must_change_password)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                username, hash_password(password), role, full_name.strip(),
                (email or "").strip() or None, int(is_active),
                student_id if role == "student" else None,
                teacher_id if role == "teacher" else None,
                int(must_change_password),
            ],
        )
    except sqlite3.IntegrityError as exc:
        if "username" in str(exc).lower():
            raise UserError(f"Username '{username}' is already taken.") from exc
        raise UserError(f"Could not create user: {exc}") from exc
    if created_by:
        db.log_activity(created_by.get("id"), created_by.get("username"), "USER_CREATED", f"{username} ({role})")
    return user_id


def list_users(role: str | None = None, search: str = "", active: str = "All") -> pd.DataFrame:
    sql = """SELECT id, username, full_name, role, email, is_active, student_id, teacher_id,
                    must_change_password, last_login, created_at
             FROM users WHERE 1=1"""
    params: list = []
    if role and role != "All":
        sql += " AND role = ?"
        params.append(role)
    if active == "Active":
        sql += " AND is_active = 1"
    elif active == "Inactive":
        sql += " AND is_active = 0"
    if search:
        sql += " AND (username LIKE ? OR full_name LIKE ? OR email LIKE ?)"
        like = f"%{search.strip()}%"
        params += [like, like, like]
    sql += " ORDER BY role, username"
    return db.query_df(sql, params)


def _active_admin_count(exclude_id: int | None = None) -> int:
    sql = "SELECT COUNT(*) FROM users WHERE role = 'admin' AND is_active = 1"
    params: list = []
    if exclude_id is not None:
        sql += " AND id != ?"
        params.append(exclude_id)
    return int(db.query_scalar(sql, params, default=0))


def set_active(user_id: int, active: bool, acting_user: dict) -> None:
    target = get_user(user_id)
    if target is None:
        raise UserError("User not found.")
    if user_id == acting_user["id"] and not active:
        raise UserError("You cannot deactivate your own account.")
    if target["role"] == "admin" and not active and _active_admin_count(exclude_id=user_id) == 0:
        raise UserError("At least one active administrator is required.")
    db.execute("UPDATE users SET is_active = ? WHERE id = ?", [int(active), user_id])
    db.log_activity(acting_user["id"], acting_user["username"],
                    "USER_ACTIVATED" if active else "USER_DEACTIVATED", target["username"])


def update_user(user_id: int, full_name: str, email: str | None, acting_user: dict) -> None:
    if not full_name or len(full_name.strip()) < 3:
        raise UserError("Full name must have at least 3 characters.")
    if email and not valid_email(email):
        raise UserError("Email address is not valid.")
    db.execute("UPDATE users SET full_name = ?, email = ? WHERE id = ?",
               [full_name.strip(), (email or "").strip() or None, user_id])
    db.log_activity(acting_user["id"], acting_user["username"], "USER_UPDATED", str(user_id))


def admin_reset_password(user_id: int, new_password: str, acting_user: dict) -> None:
    problems = password_problems(new_password)
    if problems:
        raise UserError("Password must contain " + ", ".join(problems) + ".")
    target = get_user(user_id)
    if target is None:
        raise UserError("User not found.")
    db.execute(
        """UPDATE users SET password_hash = ?, must_change_password = 1,
                            failed_attempts = 0, locked_until = NULL WHERE id = ?""",
        [hash_password(new_password), user_id],
    )
    db.log_activity(acting_user["id"], acting_user["username"], "PASSWORD_RESET", target["username"])


def unlock_user(user_id: int, acting_user: dict) -> None:
    db.execute("UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE id = ?", [user_id])
    db.log_activity(acting_user["id"], acting_user["username"], "USER_UNLOCKED", str(user_id))


def delete_user(user_id: int, acting_user: dict) -> None:
    target = get_user(user_id)
    if target is None:
        raise UserError("User not found.")
    if user_id == acting_user["id"]:
        raise UserError("You cannot delete your own account.")
    if target["role"] == "admin" and _active_admin_count(exclude_id=user_id) == 0:
        raise UserError("At least one active administrator is required.")
    db.execute("DELETE FROM users WHERE id = ?", [user_id])
    db.log_activity(acting_user["id"], acting_user["username"], "USER_DELETED", target["username"])


def recent_activity(limit: int = 50) -> pd.DataFrame:
    return db.query_df(
        "SELECT created_at, username, action, details FROM activity_log ORDER BY id DESC LIMIT ?",
        [limit],
    )
