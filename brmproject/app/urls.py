from django.urls import path
from django.views.generic import RedirectView
from django.contrib.auth.decorators import login_required
from .views.auth import login_view, register_view, logout_view, home_view, user_delete_view, user_list_view, user_update_view
from .views.transaction import transaction,transaction_view


urlpatterns = [
    # path('', RedirectView.as_view(pattern_name='userList'), name='empty_redirect'),
    path('', home_view, name='homepage'),

    path('login/', login_view, name='login'),
    path('register/', register_view, name='register'),
    path('logout/', logout_view, name='logout'),
    path('users/', user_list_view, name='user_list'),
    path('users/update/<int:user_id>/', user_update_view, name='user_update'),
    path('users/delete/<int:user_id>/', user_delete_view, name='user_delete'),
    path('transaction_view/',transaction_view, name='transaction_view'),
    path('transaction/',transaction, name='transaction'),
]
