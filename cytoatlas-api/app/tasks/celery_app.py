"""Celery application configuration."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "cytoatlas",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.process_atlas"],
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task tracking
    task_track_started=True,
    task_acks_late=True,

    # Worker settings
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_concurrency=2,  # Number of concurrent workers

    # Result settings
    result_expires=86400,  # Results expire after 24 hours

    # Task routes (optional - for scaling specific queues)
    task_routes={
        "app.tasks.process_atlas.*": {"queue": "atlas_processing"},
    },

    # Task time limits
    task_soft_time_limit=3600,  # 1 hour soft limit
    task_time_limit=7200,  # 2 hour hard limit
)


# Optional: Beat schedule for periodic tasks
celery_app.conf.beat_schedule = {
    # Example: Cleanup old jobs every day
    # "cleanup-old-jobs": {
    #     "task": "app.tasks.maintenance.cleanup_old_jobs",
    #     "schedule": 86400.0,  # 24 hours
    # },
}
