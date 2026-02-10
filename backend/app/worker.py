"""
Celery worker configuration
"""
from celery import Celery

from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "dermatology_poc",
    broker=settings.CELERY_BROKER,
    backend=settings.CELERY_BACKEND,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.PROCESSING_TIMEOUT_SECONDS * 2,
    task_soft_time_limit=settings.PROCESSING_TIMEOUT_SECONDS,
)

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.services"])


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery"""
    print(f"Request: {self.request!r}")
