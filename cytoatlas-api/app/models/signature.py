"""Signature model for cytokine/protein signatures."""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Signature(Base):
    """Cytokine/protein signature metadata table."""

    __tablename__ = "signatures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Signature identification
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    signature_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'CytoSig' or 'SecAct'

    # Description
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Gene statistics
    n_genes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    gene_list: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list

    # Category (for grouping)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<Signature(id={self.id}, name='{self.name}', type='{self.signature_type}')>"
