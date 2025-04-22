from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date
from ..models.stok import Stok
from ..models.transaksi import Transaksi
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect
from django.contrib import messages

@login_required
def transaction_view(request):
    transaksi_list = Transaksi.objects.select_related('stok_id').order_by('-created_at')
    stock_list = Stok.objects.all()
    tipe_transaksi = Transaksi.TIPE_TRANSAKSI
    return render(request, 'page/transaction.html', {'transaksi_list': transaksi_list, 'stock_list': stock_list, 'tipe_transaksi': tipe_transaksi})

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

