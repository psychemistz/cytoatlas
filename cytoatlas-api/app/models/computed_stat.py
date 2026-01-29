"""Computed statistics model for pre-aggregated analysis results."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.atlas import Atlas


class ComputedStat(Base):
    """Pre-computed statistics table for fast API responses."""

    __tablename__ = "computed_stats"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    atlas_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("atlases.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Statistic type
    stat_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'correlation', 'differential', 'activity', etc.

    # Grouping keys (what the stat is computed over)
    grouping_key1: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )  # e.g., cell_type
    grouping_key2: Mapped[str | None] = mapped_column(
        String(100), nullable=True, index=True
    )  # e.g., disease
    grouping_value1: Mapped[str | None] = mapped_column(
        String(200), nullable=True, index=True
    )  # e.g., 'CD4_T'
    grouping_value2: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )  # e.g., 'Rheumatoid Arthritis'

    # Signature information
    signature: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    signature_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'CytoSig' or 'SecAct'

    # Metric name and value
    metric: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'rho', 'pvalue', 'log2fc', 'mean_activity', etc.
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # Additional values (for multi-value stats)
    value2: Mapped[float | None] = mapped_column(Float, nullable=True)  # e.g., q-value
    value3: Mapped[float | None] = mapped_column(Float, nullable=True)  # e.g., std

    # Sample size
    n_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Additional metadata as JSON
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    atlas: Mapped["Atlas"] = relationship("Atlas", back_populates="computed_stats")

    def __repr__(self) -> str:
        return (
            f"<ComputedStat(id={self.id}, type='{self.stat_type}', "
            f"signature='{self.signature}', metric='{self.metric}')>"
        )
