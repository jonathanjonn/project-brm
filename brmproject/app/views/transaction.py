from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import date
from ..models.stok import Stok
from ..models.transaksi import Transaksi
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

@login_required
def transaction_view(request):
    if request.method == "POST":
        stok_id = request.POST.get('stok_id')
        qty = int(request.POST.get('qty'))
        tanggal_transaksi = request.POST.get('tanggal_transaksi')

        try:
            stok = Stok.objects.get(id=stok_id)
            total_harga = stok.harga * qty
            Transaksi.objects.create(
                stok_id=stok,
                qty=qty,
                total_harga=total_harga,
                tanggal_transaksi=tanggal_transaksi,
            )
            return redirect('transaction_view') 
        except Stok.DoesNotExist:
            pass

    transaksi_list = Transaksi.objects.select_related('stok_id').order_by('-created_at')
    return render(request, 'page/transaction.html', {'transaksi_list': transaksi_list})

# @csrf_exempt
@login_required
def transaction(request):
    if request.method == 'POST':
        try:
            stok_id = request.POST.get('stok_id')
            qty = int(request.POST.get('qty'))

            stok_item = Stok.objects.get(id=stok_id)

            if stok_item.stok < qty:
                return JsonResponse({'error': 'Stok tidak mencukupi'}, status=400)

            total_harga = stok_item.harga * qty

            stok_item.stok -= qty
            stok_item.save()

            transaksi = Transaksi.objects.create(
                stok_id=stok_item,
                qty=qty,
                total_harga=total_harga,
                tanggal_transaksi=date.today()
            )

            return JsonResponse({
                'message': 'Transaksi berhasil',
                'transaksi_id': transaksi.id,
                'total_harga': total_harga
            }, status=201)

        except Stok.DoesNotExist:
            return JsonResponse({'error': 'Barang tidak ditemukan'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    else:
        return JsonResponse({'error': 'Metode tidak diizinkan'}, status=405)


