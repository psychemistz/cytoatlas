"""Role-Based Access Control (RBAC) permissions system."""

from enum import Enum
from typing import Annotated

from fastapi import Depends, HTTPException, status

from app.core.security import get_current_user_optional


class Role(str, Enum):
    """User roles in the system."""

    ANONYMOUS = "anonymous"
    VIEWER = "viewer"
    RESEARCHER = "researcher"
    DATA_CURATOR = "data_curator"
    ADMIN = "admin"


class Permission(str, Enum):
    """Permissions that can be granted to roles."""

    READ_PUBLIC = "read_public"
    READ_PRIVATE = "read_private"
    WRITE_DATA = "write_data"
    MANAGE_USERS = "manage_users"
    SUBMIT_DATASET = "submit_dataset"
    USE_CHAT = "use_chat"


# Role to permissions mapping
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ANONYMOUS: {
        Permission.READ_PUBLIC,
    },
    Role.VIEWER: {
        Permission.READ_PUBLIC,
        Permission.USE_CHAT,
    },
    Role.RESEARCHER: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.USE_CHAT,
        Permission.SUBMIT_DATASET,
    },
    Role.DATA_CURATOR: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.WRITE_DATA,
        Permission.USE_CHAT,
        Permission.SUBMIT_DATASET,
    },
    Role.ADMIN: {
        Permission.READ_PUBLIC,
        Permission.READ_PRIVATE,
        Permission.WRITE_DATA,
        Permission.MANAGE_USERS,
        Permission.SUBMIT_DATASET,
        Permission.USE_CHAT,
    },
}


def get_user_permissions(role: str) -> set[Permission]:
    """Get permissions for a given role."""
    try:
        role_enum = Role(role)
        return ROLE_PERMISSIONS.get(role_enum, set())
    except ValueError:
        # Invalid role, return empty permissions
        return set()


def has_permission(user_permissions: set[Permission], required: Permission) -> bool:
    """Check if user has required permission."""
    return required in user_permissions


def require_permission(permission: Permission):
    """
    Dependency factory that creates a permission check dependency.

    Usage:
        @router.get("/protected")
        async def protected_endpoint(
            _: Annotated[None, Depends(require_permission(Permission.READ_PRIVATE))]
        ):
            ...
    """

    async def permission_checker(
        user: Annotated["User | None", Depends(get_current_user_optional)],  # type: ignore[name-defined]
    ) -> None:
        """Check if user has required permission."""
        # Determine user's role
        if user is None:
            user_role = Role.ANONYMOUS
        else:
            try:
                user_role = Role(user.role)
            except (ValueError, AttributeError):
                # Invalid role or no role attribute
                user_role = Role.ANONYMOUS

        # Get user's permissions
        user_permissions = ROLE_PERMISSIONS.get(user_role, set())

        # Check permission
        if permission not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value} required",
            )

    return permission_checker
