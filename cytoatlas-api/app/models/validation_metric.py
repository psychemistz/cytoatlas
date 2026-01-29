"""Validation metrics model for tracking inference quality."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.atlas import Atlas


class ValidationMetric(Base):
    """Validation metrics table for assessing CytoSig/SecAct inference quality."""

    __tablename__ = "validation_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    atlas_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("atlases.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Metric type
    metric_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'expression_correlation', 'gene_coverage', 'cv_stability', etc.

    # Grouping (optional)
    cell_type: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)

    # Signature information
    signature: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    signature_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'CytoSig' or 'SecAct'

    # Primary value
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # Additional statistics
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Gene coverage specific
    genes_detected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    genes_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    coverage_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Additional details as JSON (gene lists, etc.)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    atlas: Mapped["Atlas"] = relationship("Atlas", back_populates="validation_metrics")

    def __repr__(self) -> str:
        return (
            f"<ValidationMetric(id={self.id}, type='{self.metric_type}', "
            f"signature='{self.signature}', value={self.value})>"
        )
