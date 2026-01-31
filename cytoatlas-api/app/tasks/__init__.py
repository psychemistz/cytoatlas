"""Celery tasks for background processing."""

from app.tasks.celery_app import celery_app
from app.tasks.process_atlas import process_h5ad_task

__all__ = ["celery_app", "process_h5ad_task"]
