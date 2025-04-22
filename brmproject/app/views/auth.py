from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth.decorators import user_passes_test, login_required
from django.db.models import Sum
from django.utils.formats import get_format
from app.form import RegisterForm
from datetime import timedelta, date, datetime
from ..models import Transaksi
from django.http import HttpResponseRedirect
from ..models.transaksi import Transaksi
from ..models.stok import Stok

@login_required
def home_view(request):
    # Get date range from request or use default (last 30 days)
    today = date.today()
    
    # Handle date filter from user
    start_date_str = request.GET.get('start_date')
    end_date_str = request.GET.get('end_date')
    
    if start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d').date()
            
            # Enforce maximum 30 days range
            date_diff = (end_date - start_date).days
            if date_diff > 30:
                messages.error(request, "Rentang tanggal maksimal adalah 30 hari.")
                return HttpResponseRedirect(request.path)
            else:
                messages.success(request, "Filter Berhasil Dipasang")

        except ValueError:
            # Invalid date format, fallback to default
            start_date = today - timedelta(days=29)
            end_date = today
    else:
        # Default: last 30 days
        start_date = today - timedelta(days=29)
        end_date = today

    total_stok = Stok.objects.aggregate(Sum('stok'))['stok__sum'] or 0
    transaksi_harian = Transaksi.objects.filter(tanggal_transaksi=today).aggregate(Sum('qty'))['qty__sum'] or 0
    jumlah_jenis_barang = Stok.objects.count()

    # Get transactions data for the date range - using qty instead of total_harga
    transaksi_data = (
        Transaksi.objects
        .filter(tanggal_transaksi__range=[start_date, end_date])
        .values('tanggal_transaksi', 'tipe')
        .annotate(total_qty=Sum('qty'))
        .order_by('tanggal_transaksi')
    )
    
    # Prepare data for chart
    date_range = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(date_range)]
    labels = [d.strftime('%d-%m') for d in dates]
    masuk_data = [0] * date_range
    keluar_data = [0] * date_range
    
    for item in transaksi_data:
        idx = (item['tanggal_transaksi'] - start_date).days
        if item['tipe'] == 1:
            masuk_data[idx] = item['total_qty']
        elif item['tipe'] == 2:
            keluar_data[idx] = item['total_qty']

    transaksi_list = Transaksi.objects.select_related('stok_id').order_by('-created_at')[:5]
    stock_list = Stok.objects.all()
    tipe_transaksi = Transaksi.TIPE_TRANSAKSI
    
    context = {
        'labels': labels,
        'masuk_data': masuk_data,
        'keluar_data': keluar_data,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'transaksi_list': transaksi_list,
        'stock_list': stock_list,
        'tipe_transaksi': tipe_transaksi,
        'total_stok': total_stok,
        'transaksi_harian': transaksi_harian,
        'jumlah_jenis_barang': jumlah_jenis_barang
    }
    return render(request, 'page/homepage.html', context)

def login_view(request):
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, 'Login berhasil!') 
            return redirect('homepage')
        else:
            messages.error(request, 'Username atau password salah') 
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
            return redirect('user_list')
    else:
        form = RegisterForm()
    return render(request, 'auth/register.html', {'form': form})

def logout_view(request):
    if request.method == 'POST':
        logout(request)
        messages.success(request, 'Logout berhasil!')
    return redirect('login')

@user_passes_test(lambda u: u.is_superuser)
def user_list_view(request):
    users = User.objects.all()
    return render(request, 'auth/user_list.html', {'users': users})

@user_passes_test(lambda u: u.is_superuser)
def user_update_view(request, user_id):
    user = User.objects.get(pk=user_id)

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            user.username = data['username']
            if data['password']:
                user.set_password(data['password'])
            user.is_superuser = data['is_superuser']
            user.is_staff = data['is_superuser']
            user.save()
            messages.success(request, 'User berhasil diperbarui!')
            return redirect('user_list')
    else:
        form = RegisterForm(initial={
            'username': user.username,
            'is_superuser': user.is_superuser,
        })
    return render(request, 'auth/register.html', {'form': form, 'update': True, 'user_id': user.id})

@user_passes_test(lambda u: u.is_superuser)
def user_delete_view(request, user_id):
    user = User.objects.get(pk=user_id)
    if request.method == 'POST':
        if user == request.user:
            messages.error(request, 'Kamu tidak bisa menghapus akunmu sendiri!')
            return redirect('user_list')

        user.delete()
        messages.success(request, 'User berhasil dihapus!')
        return redirect('user_list')
    return render(request, 'auth/user_confirm_delete.html', {'user': user})
