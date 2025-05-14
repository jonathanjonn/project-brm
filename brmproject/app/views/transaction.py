from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date
from ..models.stok import Stok
from ..models.transaksi import Transaksi
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.core.paginator import Paginator
from django.urls import reverse

@login_required
def transaction_view(request):
    page = request.GET.get('page', 1)
    transaksi_list = Transaksi.objects.select_related('stok_id').order_by('-created_at')
    
    # Pagination - 30 data per halaman
    paginator = Paginator(transaksi_list, 30)
    transaksi_page = paginator.get_page(page)
    
    stock_list = Stok.objects.all()
    tipe_transaksi = Transaksi.TIPE_TRANSAKSI
    return render(request, 'page/transaction.html', {
        'transaksi_list': transaksi_page, 
        'stock_list': stock_list, 
        'tipe_transaksi': tipe_transaksi
    })

@login_required
def delete_transaction(request, transaction_id):
    transaction = get_object_or_404(Transaksi, id=transaction_id)
    
    stok_item = transaction.stok_id
    
    if transaction.tipe == 1:
        stok_item.stok -= transaction.qty
    elif transaction.tipe == 2:
        stok_item.stok += transaction.qty
    
    stok_item.save()
    transaction.delete()
    
    messages.success(request, f"Transaksi dengan ID {transaction_id} berhasil dihapus")
    return HttpResponseRedirect(reverse('transaction_view'))

# @csrf_exempt
@login_required
def transaction(request):
    if request.method == 'POST':
        try:
            stok_id = request.POST.get('stok_id')
            qty = int(request.POST.get('qty'))
            tipe = int(request.POST.get('tipe'))
            tanggal_transaksi = request.POST.get('tanggal_transaksi')

            stok_item = Stok.objects.get(id=stok_id)

            if stok_item.stok < qty and tipe == 2:
                return JsonResponse({'error': 'Stok tidak mencukupi'}, status=400)

            total_harga = stok_item.harga * qty

            if tipe == 1:
                stok_item.stok += qty
            elif tipe == 2:
                stok_item.stok -= qty

            stok_item.save()

            Transaksi.objects.create(
                stok_id=stok_item,
                qty=qty,
                total_harga=total_harga,
                tipe=tipe,
                tanggal_transaksi=tanggal_transaksi
            )

            messages.info(request, 'Data Berhasil Ditambahkan')
            return redirect('transaction_view')
        
        except Stok.DoesNotExist:
            return render(request, 'page/transaction.html', {
                'error': 'Barang tidak ditemukan'
            })

        except Exception as e:
            return render(request, 'page/transaction.html', {
                'error': f'Terjadi kesalahan: {str(e)}'
            })
    else:
        return render(request, 'page/transaction.html', {
                'error': f'Metode tidak di izinkan'
            })