from django.shortcuts import render
from ..models import Stok

def stock_list(request):
    query = request.GET.get("q")
    kategori = request.GET.get("kategori")

    stoks = Stok.objects.all()

    if query:
        stoks = stoks.filter(nama__icontains=query)

    if kategori:
        stoks = stoks.filter(kategori__iexact=kategori)

    kategori_list = Stok.objects.values_list('kategori', flat=True).distinct()

    return render(request, 'page/stock.html', {
        'stoks': stoks,
        'kategori_list': kategori_list,
        'selected_kategori': kategori,
        'search_query': query,
    })
