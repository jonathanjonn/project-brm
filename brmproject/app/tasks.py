from __future__ import absolute_import, unicode_literals
from celery import shared_task
from django.db import transaction
import pandas as pd
import numpy as np
import json
import logging
import math
from datetime import datetime
from django.utils import timezone


from .models import Stok, Transaksi
from .models.prediction_model import StockPredictor
from .models.prediction_result import StockPredictionResult

logger = logging.getLogger(__name__)

def round_half_up(n):
    """Fungsi pembulatan half up."""
    return int(math.floor(n + 0.5))

def calculate_days_until_stockout(forecast, current_stock):
    """Menghitung berapa hari hingga stok habis berdasarkan prediksi."""
    if current_stock <= 0:
        return 0
    if not isinstance(forecast, pd.Series):
        if isinstance(forecast, list):
            forecast = pd.Series(forecast)
        else:
            return float('inf')
            
    if all(v == 0 for v in forecast.values):
        return float('inf')
        
    cumulative_demand = 0
    for i, value in enumerate(forecast.values):
        cumulative_demand += max(0, value)
        if cumulative_demand >= current_stock:
            return i + 1
            
    return len(forecast)

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def run_prediction_for_product(self, product_id, prediction_days=30, lead_time=7):
    """
    Menjalankan proses prediksi untuk satu produk dan menyimpan hasilnya.
    """
    try:
        product = Stok.objects.get(id=product_id)
        logger.info(f"Memulai prediksi untuk produk: {product.nama} (ID: {product_id})")

        transactions = Transaksi.objects.filter(stok_id=product_id).order_by('tanggal_transaksi')
        
        min_data_points = 30
        if transactions.count() < min_data_points:
            logger.warning(f"Tidak cukup data transaksi untuk {product.nama}. Butuh minimal {min_data_points} data.")
            return f"Skipped: Not enough data for {product.nama}"

        transaction_data = [{
            'date': t.tanggal_transaksi,
            'product_id': t.stok_id.id,
            'quantity': t.qty,
            'type': t.tipe
        } for t in transactions]
        df = pd.DataFrame(transaction_data)

        predictor = StockPredictor()
        prepared_data = predictor.prepare_data(df)
        
        if prepared_data.empty or prepared_data.sum() == 0:
            logger.warning(f"Data untuk produk {product.nama} kosong atau hanya berisi nol setelah persiapan.")
            return f"Skipped: Insufficient data for {product.nama}"

        predictor.train()
        forecast_result = predictor.predict(steps=prediction_days)
        forecast_series = pd.Series(forecast_result['forecast'], index=pd.date_range(start=prepared_data.index[-1], periods=prediction_days + 1)[1:])

        reorder_info = predictor.calculate_reorder_point(lead_time)
        days_until_stockout = calculate_days_until_stockout(forecast_series, product.stok)

        # Pembulatan nilai
        forecast_values_rounded = [round_half_up(v) for v in forecast_series.values]
        reorder_point_rounded = round_half_up(reorder_info['reorder_point'])
        expected_demand_rounded = round_half_up(reorder_info['expected_demand'])
        days_until_stockout_rounded = round_half_up(days_until_stockout)

        forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast_series.index]
        forecast_data = {'dates': forecast_dates, 'values': forecast_values_rounded}

        # Simpan hasil prediksi baru setelah menghapus yang lama
        with transaction.atomic():
            # Hapus prediksi lama untuk produk ini
            StockPredictionResult.objects.filter(product=product).delete()
            
            # Buat entri prediksi baru
            StockPredictionResult.objects.create(
                product=product,
                prediction_date=timezone.now(),
                current_stock=product.stok,
                forecast_data=json.dumps(forecast_data),
                reorder_point=reorder_point_rounded,
                expected_demand=expected_demand_rounded,
                days_until_stockout=days_until_stockout_rounded,
                lead_time=lead_time,
                prediction_days=prediction_days
            )

        logger.info(f"Prediksi untuk produk {product.nama} berhasil disimpan.")
        return f"Success: Prediction for {product.nama} completed."

    except Stok.DoesNotExist:
        logger.error(f"Produk dengan ID {product_id} tidak ditemukan.")
        return f"Error: Product with ID {product_id} not found."
    except Exception as e:
        logger.error(f"Error saat prediksi untuk produk {product_id}: {str(e)}")
        self.retry(exc=e)


@shared_task
def run_all_predictions(prediction_days=30, lead_time=7):
    """
    Memicu tugas prediksi untuk semua produk yang memenuhi syarat.
    """
    product_ids = Stok.objects.values_list('id', flat=True)
    logger.info(f"Memicu prediksi untuk {len(product_ids)} produk.")
    for product_id in product_ids:
        run_prediction_for_product.delay(product_id, prediction_days, lead_time)
    return f"Triggered predictions for {len(product_ids)} products."
