"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(128), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("spotify_access_token", sa.Text(), nullable=True),
        sa.Column("spotify_refresh_token", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "videos",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("youtube_id", sa.String(20), nullable=False),
        sa.Column("title", sa.String(500), nullable=True),
        sa.Column("channel_name", sa.String(200), nullable=True),
        sa.Column("duration_secs", sa.Integer(), nullable=True),
        sa.Column("view_count", sa.Integer(), nullable=True),
        sa.Column("like_count", sa.Integer(), nullable=True),
        sa.Column("description_preview", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.JSONB(), nullable=True),
        sa.Column("thumbnail_url", sa.String(1000), nullable=True),
        sa.Column("upload_date", sa.String(20), nullable=True),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("category",
            sa.Enum("comedy","listicle","music","educational","news","review","gaming","vlog","shopping","unknown",
                    name="videocategory"),
            nullable=True,
        ),
        sa.Column("classification_confidence", sa.Float(), nullable=True),
        sa.Column("mongo_analysis_id", sa.String(24), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("youtube_id"),
    )
    op.create_index("ix_videos_youtube_id", "videos", ["youtube_id"], unique=True)

    op.create_table(
        "analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("video_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status",
            sa.Enum("queued","downloading","extracting_frames","transcribing","classifying",
                    "extracting_info","enriching","complete","failed",
                    name="jobstatus"),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("celery_task_id", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_time_secs", sa.Float(), nullable=True),
        sa.Column("mongo_result_id", sa.String(24), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["video_id"], ["videos.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analyses_status", "analyses", ["status"])

    op.create_table(
        "spotify_playlists",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("spotify_playlist_id", sa.String(100), nullable=False),
        sa.Column("playlist_url", sa.String(500), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("tracks_added", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["analysis_id"], ["analyses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("spotify_playlists")
    op.drop_table("analyses")
    op.drop_table("videos")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS videocategory")
    op.execute("DROP TYPE IF EXISTS jobstatus")
