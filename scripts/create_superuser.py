#!/usr/bin/env python
"""
Usage: python scripts/create_superuser.py
Creates an initial administrator account for the YouTube Classifier platform.
"""
import asyncio
import getpass
import sys
import uuid
import re

# Add backend directory to PYTHONPATH so imports work
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.routes.auth import _hash_password
from db.database import engine, Base, AsyncSessionFactory
from db.models import User, UserRole


def is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@]+@[^@]+\.[^@]+$", email))


async def create_superuser():
    print("=== YouTube Classifier Admin Setup ===")
    
    email = input("Admin Email: ").strip()
    if not is_valid_email(email):
        print("Error: Invalid email format.")
        sys.exit(1)

    display_name = input("Display Name [Admin User]: ").strip()
    if not display_name:
        display_name = "Admin User"

    password = getpass.getpass("Password: ")
    if len(password) < 8:
        print("Error: Password must be at least 8 characters.")
        sys.exit(1)
        
    confirm_password = getpass.getpass("Confirm Password: ")
    if password != confirm_password:
        print("Error: Passwords do not match.")
        sys.exit(1)

    async with AsyncSessionFactory() as session:
        # Check if user already exists
        existing = await session.execute(select(User).where(User.email == email))
        if existing.scalar_one_or_none():
            print(f"Error: User with email {email} already exists.")
            sys.exit(1)

        hashed_kw = _hash_password(password)
        admin_user = User(
            id=uuid.uuid4(),
            email=email,
            display_name=display_name,
            hashed_password=hashed_kw,
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
        )

        session.add(admin_user)
        await session.commit()
        
    print(f"\nSuccess! Administrator account for {email} created successfully.")


if __name__ == "__main__":
    asyncio.run(create_superuser())
