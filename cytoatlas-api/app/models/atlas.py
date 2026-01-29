"""Atlas model for tracking data sources."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.cell_type import CellType
    from app.models.computed_stat import ComputedStat
    from app.models.sample import Sample
    from app.models.validation_metric import ValidationMetric


class Atlas(Base):
    """Atlas metadata table."""

    __tablename__ = "atlases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0.0")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Data statistics
    n_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_cell_types: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # File paths
    h5ad_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="active", index=True
    )

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

    # Relationships
    samples: Mapped[list["Sample"]] = relationship(
        "Sample", back_populates="atlas", cascade="all, delete-orphan"
    )
    cell_types: Mapped[list["CellType"]] = relationship(
        "CellType", back_populates="atlas", cascade="all, delete-orphan"
    )
    computed_stats: Mapped[list["ComputedStat"]] = relationship(
        "ComputedStat", back_populates="atlas", cascade="all, delete-orphan"
    )
    validation_metrics: Mapped[list["ValidationMetric"]] = relationship(
        "ValidationMetric", back_populates="atlas", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Atlas(id={self.id}, name='{self.name}', n_cells={self.n_cells})>"
