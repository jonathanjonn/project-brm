from django.shortcuts import render
from django.db.models import F
from ..models import Stok
from django.contrib.auth.decorators import login_required

@login_required
def stock_list(request):
    search_query = request.GET.get('q', '')
    selected_kategori = request.GET.get('kategori', '')
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
        stoks = stoks.filter(kategori=selected_kategori)

    stoks = stoks.order_by(sort_expression)

    kategori_list = Stok.objects.values_list('kategori', flat=True).distinct()

    context = {
        'stoks': stoks,
        'search_query': search_query,
        'selected_kategori': selected_kategori,
        'kategori_list': kategori_list,
        'current_sort': current_sort,
        'current_order': current_order,
    }
    return render(request, 'page/stock.html', context)