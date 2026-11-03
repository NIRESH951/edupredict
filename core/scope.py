"""
Row-level access control.

A `Scope` describes which students the current user may see:
  * admin   -> every student
  * teacher -> students in the classes (department + semester + section)
               assigned to that teacher
  * student -> only their own record

Every service that reads or writes student data takes a Scope and applies
`Scope.student_filter()` to its SQL, so access rules live in one place.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.database import query_scalar


class AccessDenied(PermissionError):
    """Raised when a user tries to read or change data outside their scope."""


@dataclass(frozen=True)
class Scope:
    role: str
    user_id: int | None = None
    username: str | None = None
    student_id: str | None = None
    teacher_id: str | None = None

    @classmethod
    def from_user(cls, user: dict) -> "Scope":
        return cls(
            role=user["role"],
            user_id=user.get("id"),
            username=user.get("username"),
            student_id=user.get("student_id"),
            teacher_id=user.get("teacher_id"),
        )

    @classmethod
    def admin(cls) -> "Scope":
        return cls(role="admin", user_id=None, username="system")

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_teacher(self) -> bool:
        return self.role == "teacher"

    @property
    def is_student(self) -> bool:
        return self.role == "student"

    def student_filter(self, column: str = "student_id") -> tuple[str, list]:
        """SQL fragment + params restricting `column` to students in scope."""
        if self.is_admin:
            return "1=1", []
        if self.is_teacher:
            if not self.teacher_id:
                return "1=0", []
            return (
                f"{column} IN (SELECT s.student_id FROM students s "
                "JOIN teacher_assignments ta ON s.department = ta.department "
                "AND s.semester = ta.semester AND s.section = ta.section "
                "WHERE ta.teacher_id = ?)",
                [self.teacher_id],
            )
        if self.is_student:
            if not self.student_id:
                return "1=0", []
            return f"{column} = ?", [self.student_id]
        return "1=0", []

    def can_access_student(self, student_id: str) -> bool:
        clause, params = self.student_filter("student_id")
        count = query_scalar(
            f"SELECT COUNT(*) FROM students WHERE student_id = ? AND {clause}",
            [student_id, *params],
            default=0,
        )
        return int(count) > 0

    def ensure_student_access(self, student_id: str) -> None:
        if not self.can_access_student(student_id):
            raise AccessDenied(f"You do not have access to student {student_id}")

    @property
    def can_edit_records(self) -> bool:
        return self.role in ("admin", "teacher")
