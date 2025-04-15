from ..models import User
from django.shortcuts import render, redirect, get_object_or_404
from ..form import UserForm

def userList(request):
    users = User.objects.all()
    return render(request, 'user/user.html', {'users' : users})

def create(request):
    form = UserForm(request.POST or None)
    if form.is_valid():
        form.save()
        return redirect('userList')
    return render(request, 'user/userform.html', {'form' : form})

def update(request, pk):
    user = get_object_or_404(User, pk=pk)
    form = UserForm(request.POST or None, instance=user)
    if form.is_valid():
        form.save()
        return redirect('userList')
    return render(request, 'user/userform.html', {'form' : form})
        
    
def delete(request, pk):
    user = get_object_or_404(User, pk=pk)
    if request.method == 'POST':
        user.delete()
        return redirect('userList')
    return render(request, 'user/user.html', {'user' : user})
