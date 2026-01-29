"""Sample model for tracking individual samples/donors."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.atlas import Atlas


class Sample(Base):
    """Sample/donor metadata table."""

    __tablename__ = "samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    atlas_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("atlases.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Sample identifiers
    sample_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    donor_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Demographics
    sex: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Clinical
    disease: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    disease_group: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    condition: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Treatment
    therapy: Mapped[str | None] = mapped_column(String(200), nullable=True)
    therapy_response: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    timepoint: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Tissue
    tissue: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    organ: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Cell counts
    n_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Cohort (for multi-cohort studies)
    cohort: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Additional metadata as JSON
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    atlas: Mapped["Atlas"] = relationship("Atlas", back_populates="samples")

    def __repr__(self) -> str:
        return f"<Sample(id={self.id}, sample_id='{self.sample_id}', disease='{self.disease}')>"
