from django.urls import path
from django.views.generic import RedirectView
# from app.views import homepage, about, resume
from . import views

urlpatterns = [
    # path('', RedirectView.as_view(pattern_name='homepage'), name='empty_redirect'),

    path('index/', views.index, name='index'),
]