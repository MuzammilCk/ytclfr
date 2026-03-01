"""
api/routes/users.py

Admin-only user management endpoints.
Requires the caller to be an authenticated ADMIN.
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.auth import get_current_admin
from db.database import get_db_session
from db.models import User, UserRole
from models.schemas import (
    OKResponse,
    PaginatedResponse,
    UserResponse,
    UserRoleUpdateRequest,
    UserStatusUpdateRequest,
)

router = APIRouter(
    prefix="/api/v1/users",
    tags=["Admin Users"],
    dependencies=[Depends(get_current_admin)],
)


@router.get("", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = Query(None, description="Search by email or display name"),
    db: AsyncSession = Depends(get_db_session),
):
    """List all registered users. Admin only."""
    query = select(User)
    count_query = select(func.count(User.id))

    if search:
        search_filter = (User.email.ilike(f"%{search}%")) | (User.display_name.ilike(f"%{search}%"))
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    skip = (page - 1) * page_size
    query = query.order_by(User.created_at.desc()).offset(skip).limit(page_size)

    total = await db.scalar(count_query)
    users = (await db.scalars(query)).all()
    has_next = (skip + page_size) < total

    # Convert users to UserResponse dicts to satisfy BaseModel parsing
    user_responses = [UserResponse.model_validate(u) for u in users]

    return PaginatedResponse(
        total=total,
        page=page,
        page_size=page_size,
        has_next=has_next,
        items=user_responses,
    )


@router.patch("/{user_id}/role", response_model=UserResponse)
async def update_user_role(
    user_id: UUID,
    body: UserRoleUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_admin: User = Depends(get_current_admin),
):
    """Promote or demote a user (Admin only)."""
    if str(user_id) == str(current_admin.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot change your own role.",
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        new_role = UserRole(body.role.lower())
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role. Must be one of {[e.value for e in UserRole]}",
        )

    user.role = new_role
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/{user_id}/status", response_model=UserResponse)
async def update_user_status(
    user_id: UUID,
    body: UserStatusUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
    current_admin: User = Depends(get_current_admin),
):
    """Activate or deactivate a user account (Admin only)."""
    if str(user_id) == str(current_admin.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot deactivate yourself.",
        )

    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = body.is_active
    await db.commit()
    await db.refresh(user)
    return user
