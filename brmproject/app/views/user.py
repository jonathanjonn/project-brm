from ..models import User
from django.shortcuts import render, redirect, get_object_or_404
from ..form import UserForm
from django.contrib.auth.decorators import login_required

@login_required
def userList(request):
    users = User.objects.all()
    return render(request, 'auth/user.html', {'users' : users})

@login_required
def create(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        form.save()
        return redirect('userList')
    return render(request, 'auth/userform.html', {'form' : form})

@login_required
def update(request, pk):
    user = get_object_or_404(User, pk=pk)
    form = UserForm(request.POST or None, instance=user)
    if form.is_valid():
        form.save()
        return redirect('userList')
    return render(request, 'auth/userform.html', {'form' : form})
        
@login_required
def delete(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == 'POST':
        user.delete()
        return redirect('userList')
    return render(request, 'auth/user.html', {'user' : user})
