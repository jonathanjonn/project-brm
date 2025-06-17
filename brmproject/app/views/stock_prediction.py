from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Sum, Count, F, Q, Max
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import logging
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid requiring a display
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import traceback
import matplotlib.dates as mdates
import math

from ..models import Stok, Transaksi
from ..models.prediction_model import StockPredictor
from ..models.prediction_result import StockPredictionResult

# Setup logger
logger = logging.getLogger(__name__)

def check_new_transactions(product_id):
    """
    Mengecek apakah ada transaksi baru sejak prediksi terakhir
    
    Args:
        product_id: ID produk yang akan dicek
        
    Returns:
        tuple: (ada_transaksi_baru, tanggal_transaksi_terakhir)
    """
    try:
        # Ambil prediksi terakhir
        last_prediction = StockPredictionResult.objects.filter(
            product_id=product_id
        ).order_by('-prediction_date').first()
        
        if not last_prediction:
            return True, None  # Jika belum ada prediksi, anggap ada transaksi baru
            
        # Ambil transaksi terakhir
        last_transaction = Transaksi.objects.filter(
            stok_id=product_id
        ).order_by('-tanggal_transaksi').first()
        
        if not last_transaction:
            return False, last_prediction.prediction_date
            
        # Konversi tanggal ke datetime jika perlu
        prediction_date = last_prediction.prediction_date
        if isinstance(prediction_date, datetime):
            prediction_date = prediction_date.date()
            
        transaction_date = last_transaction.tanggal_transaksi
        if isinstance(transaction_date, datetime):
            transaction_date = transaction_date.date()
            
        # Bandingkan tanggal
        return transaction_date > prediction_date, last_transaction.tanggal_transaksi
        
    except Exception as e:
        logger.error(f"Error checking new transactions: {str(e)}")
        return True, None  # Jika error, anggap ada transaksi baru

def get_latest_prediction(product_id):
    """
    Mengambil prediksi terakhir dari database
    
    Args:
        product_id: ID produk
        
    Returns:
        dict: Data prediksi terakhir
    """
    try:
        prediction = StockPredictionResult.objects.filter(
            product_id=product_id
        ).order_by('-prediction_date').first()
        
        if not prediction:
            return None
            
        forecast_data = prediction.get_forecast()
        
        return {
            'forecast_dates': forecast_data.get('dates', []),
            'forecast_values': forecast_data.get('values', []),
            'reorder_point': prediction.reorder_point,
            'expected_demand': prediction.expected_demand,
            'days_until_stockout': prediction.days_until_stockout,
            'current_stock': prediction.current_stock,
            'prediction_date': prediction.prediction_date
        }
    except Exception as e:
        logger.error(f"Error getting latest prediction: {str(e)}")
        return None

@login_required
def stock_prediction(request):
    """View for stock prediction dashboard"""
    selected_product_id = request.GET.get('product_id')
    
    try:
        prediction_days = int(request.GET.get('days', 30))
        if prediction_days <= 0 or prediction_days > 365:
            prediction_days = 30
            messages.warning(request, "Periode prediksi harus antara 1-365 hari. Menggunakan nilai default 30 hari.")
    except (ValueError, TypeError):
        prediction_days = 30
        messages.warning(request, "Format periode prediksi tidak valid. Menggunakan nilai default 30 hari.")
    
    try:
        lead_time = int(request.GET.get('lead_time', 7))
        if lead_time <= 0 or lead_time > 90:
            lead_time = 7
            messages.warning(request, "Lead time harus antara 1-90 hari. Menggunakan nilai default 7 hari.")
    except (ValueError, TypeError):
        lead_time = 7
        messages.warning(request, "Format lead time tidak valid. Menggunakan nilai default 7 hari.")
    
    context = {
        'selected_product_id': selected_product_id,
        'prediction_days': prediction_days,
        'lead_time': lead_time,
    }
    
    if selected_product_id:
        try:
            selected_product_id = int(selected_product_id)
            selected_product = get_object_or_404(Stok, id=selected_product_id)
            context['selected_product'] = selected_product
            
            # Cek apakah ada transaksi baru
            has_new_transactions, last_transaction_date = check_new_transactions(selected_product_id)
            
            if not has_new_transactions:
                # Gunakan data dari database
                prediction_data = get_latest_prediction(selected_product_id)
                if prediction_data:
                    # Generate visualization
                    plt.figure(figsize=(10, 6))
                    plt.plot(prediction_data['forecast_dates'], 
                            prediction_data['forecast_values'], 
                            color='red', label='Prediksi')
                    
                    # Add horizontal line for reorder point
                    plt.axhline(y=prediction_data['reorder_point'], color='green', linestyle='--', 
                               label=f'Reorder Point ({prediction_data["reorder_point"]:.2f})')
                    
                    # Add current stock level
                    plt.axhline(y=selected_product.stok, color='blue', linestyle='-', 
                               label=f'Stok Saat Ini ({selected_product.stok})')
                    
                    plt.title(f'Prediksi Stok: {selected_product.nama}')
                    plt.xlabel('Tanggal')
                    plt.ylabel('Jumlah')
                    plt.legend()
                    plt.grid(True)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # Save plot to buffer
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', dpi=100)
                    buffer.seek(0)
                    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    # Generate recommendations
                    recommendations = generate_stock_recommendations(
                        current_stock=selected_product.stok,
                        reorder_point=prediction_data['reorder_point'],
                        days_until_stockout=prediction_data['days_until_stockout'],
                        expected_demand=prediction_data['expected_demand'],
                        lead_time=lead_time
                    )
                    
                    context.update({
                        'plot_data': plot_data,
                        'forecast_dates': json.dumps(prediction_data['forecast_dates']),
                        'forecast_values': json.dumps(prediction_data['forecast_values']),
                        'reorder_info': {
                            'expected_demand': prediction_data['expected_demand'],
                            'reorder_point': prediction_data['reorder_point']
                        },
                        'current_stock': selected_product.stok,
                        'days_until_stockout': prediction_data['days_until_stockout'],
                        'recommendations': recommendations,
                        'prediction_date': prediction_data['prediction_date']
                    })
                    
                    messages.info(request, f"Menggunakan prediksi terakhir dari {prediction_data['prediction_date'].strftime('%Y-%m-%d %H:%M')}")
                    return render(request, 'page/stock_prediction.html', context)
            
            # Jika ada transaksi baru, lakukan prediksi baru
            # Get all transactions for this product
            transactions = Transaksi.objects.filter(
                stok_id=selected_product_id
            ).order_by('tanggal_transaksi')
            
            # Check if we have enough data
            min_data_points = 30  # Increased from 14 to get more robust predictions
            if transactions.count() < min_data_points:
                messages.warning(
                    request, 
                    f"Tidak cukup data transaksi untuk {selected_product.nama}. "
                    f"Butuh minimal {min_data_points} hari data untuk prediksi yang akurat."
                )
                return render(request, 'page/stock_prediction.html', context)
            
            # Prepare transaction data
            transaction_data = []
            for t in transactions:
                transaction_data.append({
                    'date': t.tanggal_transaksi,
                    'product_id': t.stok_id.id,
                    'quantity': t.qty,
                    'type': t.tipe
                })
            df = pd.DataFrame(transaction_data)
            
            # Initialize predictor
            predictor = StockPredictor()
            
            # Train model
            try:
                # Prepare data
                predictor.prepare_data(df, product_id=selected_product_id)
                
                # Check if data is too sparse (many zeros)
                zero_ratio = (predictor.history_data == 0).mean()
                if zero_ratio > 0.7:  # If more than 70% of data points are zero
                    messages.warning(
                        request,
                        f"Data transaksi untuk {selected_product.nama} terlalu jarang. "
                        f"Prediksi mungkin kurang akurat ({int(zero_ratio*100)}% data adalah nol)."
                    )
                
                # Check for stationarity and apply differencing if needed
                is_stationary = predictor.check_stationarity(predictor.history_data)
                
                # Find optimal parameters using grid search or auto ARIMA
                params = predictor.find_optimal_parameters(
                    max_p=2, max_q=2, max_d=1, 
                    seasonal_period=7  # Weekly seasonality
                )
                
                # Cross-validate model to test performance
                validation_result = None
                if len(predictor.history_data) >= 60:  # Only if we have enough data
                    validation_result = predictor.validate(test_size=min(30, int(len(predictor.history_data) * 0.3)))
                    context['validation_metrics'] = {
                        'rmse': round(float(validation_result['rmse']), 2),
                        'mape': round(float(validation_result['mape']), 2)
                    }
                
                # Train final model on all data
                model = predictor.train()
                
                # Make predictions
                forecast_result = predictor.predict(steps=prediction_days)
                forecast = forecast_result['forecast']
                confidence_intervals = forecast_result['confidence_intervals']
                
                # Calculate reorder point with dynamic safety factor
                safety_level = 0.95  # 95% service level
                reorder_info = predictor.calculate_reorder_point(lead_time, service_level=safety_level)
                
                # Generate visualization
                plt.figure(figsize=(10, 6))
                history_to_show = min(90, len(predictor.history_data))
                plt.plot(predictor.history_data[-history_to_show:].index, 
                         predictor.history_data[-history_to_show:], 
                         label='Data Historis')
                plt.plot(forecast.index, forecast, color='red', label='Prediksi')
                plt.fill_between(
                    confidence_intervals.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1], 
                    color='pink', alpha=0.3,
                    label='Interval Kepercayaan 95%'
                )
                
                # Add horizontal line for reorder point
                plt.axhline(y=reorder_info['reorder_point'], color='green', linestyle='--', 
                           label=f'Reorder Point ({reorder_info["reorder_point"]:.2f})')
                
                # Add current stock level
                plt.axhline(y=selected_product.stok, color='blue', linestyle='-', 
                           label=f'Stok Saat Ini ({selected_product.stok})')
                
                plt.title(f'Prediksi Stok: {selected_product.nama}')
                plt.xlabel('Tanggal')
                plt.ylabel('Jumlah')
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save plot to buffer
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                # Format forecast data for template
                forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast.index]
                forecast_values = [round(float(v), 2) for v in forecast.values]
                
                # Format reorder information
                reorder_info = {
                    'expected_demand': round(float(reorder_info['expected_demand']), 2),
                    'reorder_point': round(float(reorder_info['reorder_point']), 2)
                }
                
                # Calculate days until stockout more precisely
                days_until_stockout = calculate_days_until_stockout(forecast, selected_product.stok)
                
                # Generate recommendations
                recommendations = generate_stock_recommendations(
                    current_stock=selected_product.stok,
                    reorder_point=reorder_info['reorder_point'],
                    days_until_stockout=days_until_stockout,
                    expected_demand=reorder_info['expected_demand'],
                    lead_time=lead_time
                )
                
                # Fungsi pembulatan half up
                def round_half_up(n):
                    return int(math.floor(n + 0.5))
                
                # Bulatkan forecast_values, reorder_point, expected_demand, days_until_stockout
                forecast_values_rounded = [round_half_up(v) for v in forecast_values]
                reorder_point_rounded = round_half_up(reorder_info['reorder_point'])
                expected_demand_rounded = round_half_up(reorder_info['expected_demand'])
                days_until_stockout_rounded = round_half_up(days_until_stockout)
                
                # Simpan hasil prediksi ke database
                forecast_data = {
                    'dates': forecast_dates,
                    'values': forecast_values_rounded
                }
                
                prediction_result = StockPredictionResult(
                    product=selected_product,
                    prediction_date=datetime.now(),
                    current_stock=selected_product.stok,
                    forecast_data=json.dumps(forecast_data),
                    reorder_point=reorder_point_rounded,
                    expected_demand=expected_demand_rounded,
                    days_until_stockout=days_until_stockout_rounded
                )
                prediction_result.save()
                
                context.update({
                    'plot_data': plot_data,
                    'forecast_dates': json.dumps(forecast_dates),
                    'forecast_values': json.dumps(forecast_values_rounded),
                    'reorder_info': reorder_info,
                    'current_stock': selected_product.stok,
                    'days_until_stockout': days_until_stockout_rounded,
                    'recommendations': recommendations,
                    'prediction_date': datetime.now()
                })
                
            except Exception as e:
                logger.error(f"Error during prediction for product {selected_product_id}: {str(e)}")
                logger.error(traceback.format_exc())
                messages.error(request, f"Error saat melakukan prediksi: {str(e)}")
        
        except Exception as e:
            logger.error(f"General error in stock prediction view: {str(e)}")
            logger.error(traceback.format_exc())
            messages.error(request, f"Error: {str(e)}")
    
    return render(request, 'page/stock_prediction.html', context)

def calculate_days_until_stockout(forecast, current_stock):
    """Calculate how many days until stock is depleted based on forecast"""
    if current_stock <= 0:
        return 0
    
    # Handle edge case when forecast values are all zero
    if all(v == 0 for v in forecast.values):
        return float('inf')  # Stock will never deplete
        
    cumulative_demand = 0
    for i, value in enumerate(forecast.values):
        cumulative_demand += max(0, value)  # Ensure non-negative demand
        if cumulative_demand >= current_stock:
            return i + 1
            
    return len(forecast)  # Stock will last beyond forecast period

def generate_stock_recommendations(current_stock, reorder_point, days_until_stockout, expected_demand, lead_time):
    """Generate recommendations based on stock levels and forecast"""
    recommendations = []
    
    if current_stock <= 0:
        recommendations.append({
            'level': 'danger',
            'message': 'Stok sudah habis! Perlu segera melakukan pembelian.'
        })
    elif current_stock < reorder_point:
        recommended_order = max(reorder_point * 1.5, expected_demand * 2)
        recommendations.append({
            'level': 'warning',
            'message': f'Stok di bawah reorder point! Segera lakukan pemesanan minimal {round(recommended_order)} unit.'
        })
    elif days_until_stockout <= lead_time * 1.5:
        recommendations.append({
            'level': 'info',
            'message': f'Stok akan habis dalam {days_until_stockout} hari, yang dekat dengan lead time. '
                      f'Pertimbangkan untuk melakukan pemesanan dalam waktu dekat.'
        })
    else:
        recommendations.append({
            'level': 'success',
            'message': f'Stok mencukupi untuk {days_until_stockout} hari ke depan.'
        })
    
    return recommendations

@login_required
def get_product_history(request, product_id):
    """AJAX endpoint to get product transaction history"""
    try:
        product = get_object_or_404(Stok, id=product_id)
        transactions = Transaksi.objects.filter(
            stok_id=product_id
        ).order_by('-tanggal_transaksi')[:60]  # Last 60 transactions for better overview
        
        data = []
        for t in transactions:
            data.append({
                'date': t.tanggal_transaksi.strftime('%Y-%m-%d'),
                'qty': t.qty,
                'type': t.tipe,
                'total': t.total_harga if hasattr(t, 'total_harga') else 0
            })
            
        # Calculate additional statistics
        stats = {
            'total_transactions': transactions.count(),
            'avg_quantity': float(transactions.aggregate(avg=Sum('qty')/Count('id'))['avg']) if transactions else 0,
            'last_transaction': transactions.first().tanggal_transaksi.strftime('%Y-%m-%d') if transactions else None
        }
            
        return JsonResponse({'status': 'success', 'data': data, 'stats': stats}, safe=True)
    except Exception as e:
        logger.error(f"Error in get_product_history: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full traceback
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@login_required
def stock_prediction_all(request):
    """View untuk menampilkan prediksi stok semua produk"""
    try:
        prediction_days = int(request.GET.get('days', 30))
        if prediction_days <= 0 or prediction_days > 365:
            prediction_days = 30
            messages.warning(request, "Periode prediksi harus antara 1-365 hari. Menggunakan nilai default 30 hari.")
    except (ValueError, TypeError):
        prediction_days = 30
        messages.warning(request, "Format periode prediksi tidak valid. Menggunakan nilai default 30 hari.")
    
    try:
        lead_time = int(request.GET.get('lead_time', 7))
        if lead_time <= 0 or lead_time > 90:
            lead_time = 7
            messages.warning(request, "Lead time harus antara 1-90 hari. Menggunakan nilai default 7 hari.")
    except (ValueError, TypeError):
        lead_time = 7
        messages.warning(request, "Format lead time tidak valid. Menggunakan nilai default 7 hari.")
    
    status_filter = request.GET.get('status_filter', '')
    
    # Get all products
    products = Stok.objects.all().order_by('nama')
    products_with_predictions = []
    
    for product in products:
        try:
            # Cek apakah ada transaksi baru
            has_new_transactions, last_transaction_date = check_new_transactions(product.id)
            
            if not has_new_transactions:
                # Gunakan data dari database
                prediction_data = get_latest_prediction(product.id)
                if prediction_data:
                    # Generate recommendations
                    recommendations = generate_stock_recommendations(
                        current_stock=product.stok,
                        reorder_point=prediction_data['reorder_point'],
                        days_until_stockout=prediction_data['days_until_stockout'],
                        expected_demand=prediction_data['expected_demand'],
                        lead_time=lead_time
                    )
                    
                    # Determine status
                    status_class = 'success'
                    status_text = 'Stok Aman'
                    
                    if product.stok <= 0:
                        status_class = 'danger'
                        status_text = 'Stok Habis'
                    elif product.stok <= prediction_data['reorder_point']:
                        status_class = 'warning'
                        status_text = 'Perlu Reorder'
                    
                    # Apply status filter
                    if status_filter and status_class != status_filter:
                        continue
                    
                    # Add product with prediction data
                    products_with_predictions.append({
                        'id': product.id,
                        'nama': product.nama,
                        'stok': product.stok,
                        'reorder_point': prediction_data['reorder_point'],
                        'days_until_stockout': prediction_data['days_until_stockout'],
                        'recommendations': recommendations,
                        'status_class': status_class,
                        'status_text': status_text,
                        'prediction_date': prediction_data['prediction_date']
                    })
                    continue
            
            # Jika ada transaksi baru, lakukan prediksi baru
            # Get transactions for this product
            transactions = Transaksi.objects.filter(
                stok_id=product.id
            ).order_by('tanggal_transaksi')
            
            if transactions.count() < 30:  # Skip products with insufficient data
                continue
                
            # Prepare transaction data
            transaction_data = []
            for t in transactions:
                transaction_data.append({
                    'date': t.tanggal_transaksi,
                    'product_id': t.stok_id.id,
                    'quantity': t.qty,
                    'type': t.tipe
                })
            df = pd.DataFrame(transaction_data)
            
            # Initialize predictor
            predictor = StockPredictor()
            
            # Prepare and train model
            predictor.prepare_data(df, product_id=product.id)
            model = predictor.train()
            
            # Make predictions
            forecast_result = predictor.predict(steps=prediction_days)
            forecast = forecast_result['forecast']
            confidence_intervals = forecast_result['confidence_intervals']
            
            # Calculate reorder point
            reorder_info = predictor.calculate_reorder_point(lead_time)
            
            # Calculate days until stockout
            days_until_stockout = calculate_days_until_stockout(forecast, product.stok)
            
            # Generate recommendations
            recommendations = generate_stock_recommendations(
                current_stock=product.stok,
                reorder_point=reorder_info['reorder_point'],
                days_until_stockout=days_until_stockout,
                expected_demand=reorder_info['expected_demand'],
                lead_time=lead_time
            )
            
            # Determine status
            status_class = 'success'
            status_text = 'Stok Aman'
            
            if product.stok <= 0:
                status_class = 'danger'
                status_text = 'Stok Habis'
            elif product.stok <= reorder_info['reorder_point']:
                status_class = 'warning'
                status_text = 'Perlu Reorder'
            
            # Apply status filter
            if status_filter and status_class != status_filter:
                continue
            
            # Format forecast data for template
            forecast_dates = [d.strftime('%Y-%m-%d') for d in forecast.index]
            forecast_values = [round(float(v), 2) for v in forecast.values]
            
            # Fungsi pembulatan half up
            def round_half_up(n):
                return int(math.floor(n + 0.5))
            
            # Bulatkan forecast_values, reorder_point, expected_demand, days_until_stockout
            forecast_values_rounded = [round_half_up(v) for v in forecast_values]
            reorder_point_rounded = round_half_up(reorder_info['reorder_point'])
            expected_demand_rounded = round_half_up(reorder_info['expected_demand'])
            days_until_stockout_rounded = round_half_up(days_until_stockout)
            
            # Simpan hasil prediksi ke database
            forecast_data = {
                'dates': forecast_dates,
                'values': forecast_values_rounded
            }
            
            prediction_result = StockPredictionResult(
                product=product,
                prediction_date=datetime.now(),
                current_stock=product.stok,
                forecast_data=json.dumps(forecast_data),
                reorder_point=reorder_point_rounded,
                expected_demand=expected_demand_rounded,
                days_until_stockout=days_until_stockout_rounded
            )
            prediction_result.save()
            
            # Add product with prediction data
            products_with_predictions.append({
                'id': product.id,
                'nama': product.nama,
                'stok': product.stok,
                'reorder_point': reorder_point_rounded,
                'days_until_stockout': days_until_stockout_rounded,
                'recommendations': recommendations,
                'status_class': status_class,
                'status_text': status_text,
                'prediction_date': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Error processing product {product.id}: {str(e)}")
            continue
    
    context = {
        'products': products_with_predictions,
        'prediction_days': prediction_days,
        'lead_time': lead_time,
        'status_filter': status_filter
    }
    
    return render(request, 'page/stock_prediction_all.html', context)