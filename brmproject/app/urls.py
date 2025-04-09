from django.urls import path
from django.views.generic import RedirectView
from app.views import homepage, about, resume

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='homepage'), name='empty_redirect'),

    path('homepage/', homepage.index, name='homepage'),
]