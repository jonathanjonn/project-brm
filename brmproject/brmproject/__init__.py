
# Ini akan memastikan app diimpor ketika
# Django dimulai sehingga @shared_task akan menggunakan app ini.
from .celery import app as celery_app

__all__ = ('celery_app',)
