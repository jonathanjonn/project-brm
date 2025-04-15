from django.shortcuts import redirect
from django.conf import settings
from django.urls import reverse

class LoginRequiredMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.allowed_paths = [
            reverse('login'),
            reverse('register'),
        ]

    def __call__(self, request):
        if not request.user.is_authenticated and request.path not in self.allowed_paths:
            return redirect('login')
        return self.get_response(request)
