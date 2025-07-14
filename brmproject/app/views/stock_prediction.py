from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.urls import reverse
from datetime import datetime
import json
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from django.db.models import Sum, Count

from ..models import Stok, Transaksi, StockPredictionResult
from ..tasks import run_prediction_for_product, run_all_predictions

logger = logging.getLogger(__name__)

def get_latest_prediction(product_id):
    return StockPredictionResult.objects.filter(product_id=product_id).order_by('-prediction_date').first()

def check_new_transactions(product_id):
    try:
        last_prediction = get_latest_prediction(product_id)
        if not last_prediction:
            return True
        last_transaction = Transaksi.objects.filter(stok_id=product_id).order_by('-tanggal_transaksi').first()
        if not last_transaction:
            return False
        return last_transaction.tanggal_transaksi.date() > last_prediction.prediction_date.date()
    except Exception as e:
        logger.error(f"Error checking new transactions for product {product_id}: {e}")
        return True

def generate_stock_recommendations(current_stock, reorder_point, days_until_stockout, expected_demand, lead_time):
    recommendations = []
    if current_stock <= 0:
        recommendations.append({'level': 'danger', 'message': 'Stok habis! Perlu segera restock.'})
    elif current_stock < reorder_point:
        recommended_order = max(reorder_point * 1.5, expected_demand * 2)
        recommendations.append({'level': 'warning', 'message': f'Stok di bawah reorder point! Segera pesan min. {round(recommended_order)} unit.'})
    elif days_until_stockout is not None and days_until_stockout <= lead_time * 1.5:
        recommendations.append({'level': 'info', 'message': f'Stok akan habis dalam {days_until_stockout} hari. Pertimbangkan memesan.'})
    else:
        recommendations.append({'level': 'success', 'message': 'Stok dalam kondisi aman.'})
    return recommendations

def get_prediction_context(prediction, product):
    context = {}
    try:
        forecast_data = prediction.get_forecast()
        dates = forecast_data.get('dates', [])
        values = forecast_data.get('values', [])

        plt.figure(figsize=(10, 6))
        plt.plot(dates, values, color='red', label='Prediksi')
        plt.axhline(y=prediction.reorder_point, color='green', linestyle='--', label=f'Reorder Point ({prediction.reorder_point})')
        plt.axhline(y=product.stok, color='blue', linestyle='-', label=f'Stok Saat Ini ({product.stok})')
        plt.title(f'Prediksi Stok: {product.nama}')
        plt.xlabel('Tanggal'); plt.ylabel('Jumlah'); plt.legend(); plt.grid(True); plt.xticks(rotation=45, ha='right'); plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        context.update({
            'prediction_date': prediction.prediction_date,
            'current_stock': prediction.current_stock,
            'reorder_info': {'reorder_point': prediction.reorder_point, 'expected_demand': prediction.expected_demand},
            'days_until_stockout': prediction.days_until_stockout,
            'plot_data': plot_data,
            'forecast_dates': json.dumps(dates),
            'forecast_values': json.dumps(values),
            'recommendations': generate_stock_recommendations(prediction.current_stock, prediction.reorder_point, prediction.days_until_stockout, prediction.expected_demand, prediction.lead_time),
        })
    except Exception as e:
        logger.error(f"Error generating prediction context: {e}")
    return context

@login_required
def stock_prediction_all(request):
    # This view remains the same as the last correct version
    try:
        prediction_days = int(request.GET.get('days', 30))
        lead_time = int(request.GET.get('lead_time', 7))
        status_filter = request.GET.get('status_filter', '')

        run_all_predictions.delay(prediction_days, lead_time)
        messages.info(request, f"Memperbarui prediksi untuk semua produk dengan periode {prediction_days} hari dan lead time {lead_time} hari.")
        
        products = Stok.objects.all().order_by('nama')
        products_with_predictions = []
        for product in products:
            latest_prediction = get_latest_prediction(product.id)
            product_data = {'id': product.id, 'nama': product.nama, 'stok': product.stok}
            if latest_prediction:
                status = latest_prediction.status
                if status_filter and status != status_filter:
                    continue
                product_data.update({
                    'reorder_point': latest_prediction.reorder_point,
                    'days_until_stockout': latest_prediction.days_until_stockout,
                    'prediction_date': latest_prediction.prediction_date,
                    'recommendations': generate_stock_recommendations(latest_prediction.current_stock, latest_prediction.reorder_point, latest_prediction.days_until_stockout, latest_prediction.expected_demand, latest_prediction.lead_time),
                    'status_class': status,
                    'status_text': latest_prediction.status_text,
                })
            else:
                if status_filter:
                    continue
                product_data.update({'status_class': 'secondary', 'status_text': 'Belum Ada Prediksi'})
            products_with_predictions.append(product_data)

        context = {
            'products': products_with_predictions,
            'status_filter': status_filter,
            'prediction_days': prediction_days,
            'lead_time': lead_time,
        }
        return render(request, 'page/stock_prediction_all.html', context)
    except (ValueError, TypeError):
        messages.error(request, "Input tidak valid.")
        return redirect(reverse('stock_prediction_all'))

@login_required
def stock_prediction(request):
    selected_product_id = request.GET.get('product_id')
    context = {'products': Stok.objects.all().order_by('nama')}

    if not selected_product_id:
        messages.info(request, "Silakan pilih produk untuk memulai analisis.")
        return render(request, 'page/stock_prediction.html', context)

    try:
        product = get_object_or_404(Stok, id=selected_product_id)
        context['selected_product'] = product
        context['selected_product_id'] = selected_product_id

        prediction_days = int(request.GET.get('days', 30))
        lead_time = int(request.GET.get('lead_time', 7))
        context.update({'prediction_days': prediction_days, 'lead_time': lead_time})

        force_predict = request.GET.get('predict', 'false').lower() == 'true'
        has_new_transactions = check_new_transactions(selected_product_id)
        latest_prediction = get_latest_prediction(selected_product_id)

        if force_predict or has_new_transactions or not latest_prediction:
            run_prediction_for_product.delay(selected_product_id, prediction_days, lead_time)
            messages.info(request, f"Prediksi untuk {product.nama} sedang diproses. Halaman akan diperbarui secara otomatis.")

        if latest_prediction:
            context.update(get_prediction_context(latest_prediction, product))
        else:
            messages.warning(request, f"Belum ada data prediksi untuk {product.nama}. Proses sedang berjalan jika baru dipicu.")

    except (ValueError, TypeError):
        messages.error(request, "Product ID atau parameter tidak valid.")
        return redirect(reverse('stock_prediction'))
    except Exception as e:
        logger.error(f"Error di view stock_prediction: {e}")
        messages.error(request, "Terjadi kesalahan saat memproses permintaan Anda.")
        
    return render(request, 'page/stock_prediction.html', context)

@login_required
def get_product_history(request, product_id):
    try:
        transactions = Transaksi.objects.filter(stok_id=product_id).order_by('-tanggal_transaksi')[:60]
        data = [{
            'date': t.tanggal_transaksi.strftime('%Y-%m-%d'),
            'qty': t.qty,
            'type': t.tipe,
            'total': t.total_harga if hasattr(t, 'total_harga') else 0
        } for t in transactions]
        
        stats = {
            'total_transactions': transactions.count(),
            'avg_quantity': round(transactions.aggregate(avg=Sum('qty'))['avg'] or 0, 2),
            'last_transaction': transactions.first().tanggal_transaksi.strftime('%Y-%m-%d') if transactions else None
        }
        return JsonResponse({'status': 'success', 'data': data, 'stats': stats})
    except Exception as e:
        logger.error(f"Error in get_product_history for product {product_id}: {e}")
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
