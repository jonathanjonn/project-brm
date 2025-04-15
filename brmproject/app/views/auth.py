from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import user_passes_test
from django.contrib.auth.decorators import login_required

@login_required
def home_view(request):
    return render(request, 'page/homepage.html')

# LOGIN VIEW
def login_view(request):
    form = AuthenticationForm(request, data=request.POST or None)
    message = ''
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('homepage')  # Ganti sesuai URL tujuan setelah login
        else:
            message = 'Username atau password salah'
    return render(request, 'auth/login.html', {'form': form, 'message': message})

# REGISTER VIEW (hanya untuk superuser)
@user_passes_test(lambda u: u.is_superuser)
def register_view(request):
    form = UserCreationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('login')  # atau ke halaman admin/user list
    return render(request, 'auth/register.html', {'form': form})


def logout_view(request):
    if request.method == 'POST':
        logout(request)
    return redirect('login')