from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from app.form import LoginForm

def login_view(request):
    form = LoginForm(request.POST or None)
    message = ''
    if request.method == 'POST':
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('userList') 
            else:
                message = 'Username atau password salah'
    return render(request, 'user/login.html', {'form': form, 'message': message})
