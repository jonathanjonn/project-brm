from django.urls import path
from django.views.generic import RedirectView
from django.contrib.auth.decorators import login_required
from .views.auth import login_view, register_view, logout_view, home_view


urlpatterns = [
    # path('', RedirectView.as_view(pattern_name='userList'), name='empty_redirect'),
    path('', home_view, name='homepage'),

    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
]
