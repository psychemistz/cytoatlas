"""Job model for tracking H5AD processing tasks."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Job(Base):
    """Job tracking table for H5AD processing."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # User association
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Atlas information
    atlas_name: Mapped[str] = mapped_column(String(100), nullable=False)
    atlas_description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status tracking
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", index=True
    )  # pending, validating, processing, completed, failed, cancelled
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # 0-100
    current_step: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # File paths
    h5ad_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Processing metadata
    n_cells: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_cell_types: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signature_types: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # JSON array: ["CytoSig", "SecAct"]

    # Celery task tracking
    celery_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", backref="jobs")

    def __repr__(self) -> str:
        return f"<Job(id={self.id}, atlas_name='{self.atlas_name}', status='{self.status}')>"
