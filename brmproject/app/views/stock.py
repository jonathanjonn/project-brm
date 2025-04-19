from django.shortcuts import render, redirect
from django.db.models import F
from ..models import Stok
from django.contrib.auth.decorators import login_required

@login_required
def stock_list(request):
    search_query = request.GET.get('q', '')
    selected_kategori = request.GET.get('kategori', '').lower()
    current_sort = request.GET.get('sort', 'nama') 
    current_order = request.GET.get('order', 'asc')

    allowed_sort_fields = ['nama', 'kategori', 'harga', 'stok', 'created_at']
    if current_sort not in allowed_sort_fields:
        current_sort = 'nama'

    sort_expression = f"-{current_sort}" if current_order == 'desc' else current_sort

    stoks = Stok.objects.all()

    if search_query:
        stoks = stoks.filter(nama__icontains=search_query)

    if selected_kategori:
        stoks = stoks.filter(kategori__iexact=selected_kategori)

    stoks = stoks.order_by(sort_expression)

    kategori_list_raw = Stok.objects.values_list('kategori', flat=True)
    kategori_list = sorted(set(k.lower() for k in kategori_list_raw if k))

    context = {
        'stoks': stoks,
        'search_query': search_query,
        'selected_kategori': selected_kategori,
        'kategori_list': kategori_list,
        'current_sort': current_sort,
        'current_order': current_order,
    }
    return render(request, 'page/stock.html', context)



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