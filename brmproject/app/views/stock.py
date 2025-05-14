from django.shortcuts import render, redirect, get_object_or_404
from django.db.models import F
from django.core.paginator import Paginator
from ..models import Stok
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.urls import reverse

@login_required
def stock_list(request):
    search_query = request.GET.get('q', '')
    selected_kategori = request.GET.get('kategori', '').lower()
    current_sort = request.GET.get('sort', 'nama') 
    current_order = request.GET.get('order', 'asc')
    page = request.GET.get('page', 1)

    allowed_sort_fields = ['nama', 'kategori', 'harga', 'stok', 'updated_at']
    if current_sort not in allowed_sort_fields:
        current_sort = 'nama'

    sort_expression = f"-{current_sort}" if current_order == 'desc' else current_sort

    stoks = Stok.objects.all()

    if search_query:
        stoks = stoks.filter(nama__icontains=search_query)
        messages.success(request, "Filter Berhasil Dipasang")

    if selected_kategori:
        stoks = stoks.filter(kategori__iexact=selected_kategori)
        messages.success(request, "Filter Berhasil Dipasang")

    stoks = stoks.order_by(sort_expression)
    
    paginator = Paginator(stoks, 30)
    stoks_page = paginator.get_page(page)

    kategori_list_raw = Stok.objects.values_list('kategori', flat=True)
    kategori_list = sorted(set(k.lower() for k in kategori_list_raw if k))

    context = {
        'stoks': stoks_page,
        'search_query': search_query,
        'selected_kategori': selected_kategori,
        'kategori_list': kategori_list,
        'current_sort': current_sort,
        'current_order': current_order,
    }
    return render(request, 'page/stock.html', context)

@login_required
def delete_stock(request, stock_id):
    stock = get_object_or_404(Stok, id=stock_id)
    stock.delete()
    messages.success(request, f"Stok '{stock.nama}' berhasil dihapus")
    return HttpResponseRedirect(reverse('stock_list'))

@login_required
def create_stock(request):
    if request.method == "POST":
        nama = request.POST.get("nama")
        harga = request.POST.get("harga")
        kategori = request.POST.get("kategori")
        stok = request.POST.get("stok")
        
        if not all([nama, harga, kategori, stok]):
            raise ValueError('Data untuk membuat stok baru tidak lengkap')

        try:
            harga = int(harga)
            stok = int(stok)
        except ValueError:
            raise ValueError('Harga dan stok awal harus berupa angka')

        Stok.objects.create(
            nama=nama,
            harga=harga,
            kategori=kategori,
            stok=stok
        )
        
        return redirect('transaction_view')
    return render(request, 'page/create_stock.html')