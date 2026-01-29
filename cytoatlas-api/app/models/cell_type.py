"""Cell type model for tracking cell populations."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.atlas import Atlas


class CellType(Base):
    """Cell type metadata table."""

    __tablename__ = "cell_types"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    atlas_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("atlases.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Cell type names
    name: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    harmonized_name: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)

    # Hierarchy
    lineage: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    parent_type: Mapped[str | None] = mapped_column(String(200), nullable=True)

    # Statistics
    n_cells: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Optional: organ/tissue association
    organ: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    atlas: Mapped["Atlas"] = relationship("Atlas", back_populates="cell_types")

    def __repr__(self) -> str:
        return f"<CellType(id={self.id}, name='{self.name}', n_cells={self.n_cells})>"
