"""Conversation and Message models for chat functionality."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.user import User


class Conversation(Base):
    """Chat conversation table."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # User association (nullable for anonymous users)
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # Session tracking for anonymous users
    session_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # UUID

    # Conversation metadata
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Context data for download capability (JSON)
    context_data: Mapped[str | None] = mapped_column(Text, nullable=True)

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
    user: Mapped["User | None"] = relationship("User", backref="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan",
        order_by="Message.created_at"
    )

    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, title='{self.title}', session_id='{self.session_id}')>"


class Message(Base):
    """Chat message table."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Conversation association
    conversation_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Message content
    role: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # user, assistant, tool, system
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Tool usage tracking (JSON arrays)
    tool_calls: Mapped[str | None] = mapped_column(Text, nullable=True)
    tool_results: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Visualization configs (JSON array)
    visualizations: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Downloadable data reference (JSON)
    downloadable_data: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Token usage tracking
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role='{self.role}', conversation_id={self.conversation_id})>"
