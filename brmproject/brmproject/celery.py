
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery

# Atur variabel environment default untuk Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'brmproject.settings')

app = Celery('brmproject')

# Menggunakan string di sini berarti worker tidak perlu serialize
# objek konfigurasi ke child processes.
# - namespace='CELERY' berarti semua kunci konfigurasi Celery
#   harus memiliki awalan `CELERY_`.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Muat modul tasks dari semua aplikasi Django yang terdaftar.
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
