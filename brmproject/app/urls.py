from django.urls import path
from django.views.generic import RedirectView
# from app.views import homepage, about, resume
from app.views import user

urlpatterns = [
    path('', RedirectView.as_view(pattern_name='userList'), name='empty_redirect'),

    path('user/', user.userList, name='userList'),
    path('create/user/', user.create, name='create'),
    path('delete/user/<int:pk>/', user.delete, name='delete'),
    path('update/user/<int:pk>/', user.update, name='update'),
]