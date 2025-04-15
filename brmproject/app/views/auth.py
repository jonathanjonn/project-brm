from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import user_passes_test, login_required
from app.form import RegisterForm

@login_required
def home_view(request):
    return render(request, 'page/homepage.html')

def login_view(request):
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, 'Login berhasil!')  # Pesan sukses setelah login
            return redirect('homepage')
        else:
            messages.error(request, 'Username atau password salah')  # Pesan error jika gagal login
    return render(request, 'auth/login.html', {'form': form})

@user_passes_test(lambda u: u.is_superuser)
def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            user = User.objects.create_user(
                username=data['username'],
                password=data['password']
            )
            user.is_superuser = data['is_superuser']
            user.is_staff = data['is_superuser']
            user.save()
            messages.success(request, 'User berhasil didaftarkan!')
            return redirect('register')  # Ganti dengan URL tujuan setelah register
    else:
        form = RegisterForm()
    return render(request, 'auth/register.html', {'form': form})

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        messages.success(request, 'Logout berhasil!')  # Pesan sukses setelah logout
    return redirect('login')
