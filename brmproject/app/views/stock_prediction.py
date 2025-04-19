from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db.models import Sum, Count
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid requiring a display
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from ..models import Stok, Transaksi
from ..models.prediction_model import StockPredictor

@login_required
def stock_prediction(request):
    """View for stock prediction dashboard"""
    products = Stok.objects.all().order_by('nama')
    selected_product_id = request.GET.get('product_id')
    prediction_days = int(request.GET.get('days', 30))
    lead_time = int(request.GET.get('lead_time', 7))
    
    context = {
        'products': products,
        'selected_product_id': selected_product_id,
        'prediction_days': prediction_days,
        'lead_time': lead_time,
    }
    
    if selected_product_id:
        try:
            selected_product_id = int(selected_product_id)
            selected_product = get_object_or_404(Stok, id=selected_product_id)
            context['selected_product'] = selected_product
            
            # Get all transactions for this product
            transactions = Transaksi.objects.filter(
                stok_id=selected_product_id
            ).order_by('tanggal_transaksi')
            
            if transactions.count() < 14:  # Need at least 2 weeks of data
                messages.warning(
                    request, 
                    f"Tidak cukup data transaksi untuk {selected_product.nama}. Butuh minimal 14 hari data."
                )
                return render(request, 'page/stock_prediction.html', context)
            
            # Prepare transaction data
            transaction_data = []
            for t in transactions:
                transaction_data.append({
                    'date': t.tanggal_transaksi,
                    'product_id': t.stok_id.id,
                    'quantity': t.qty
                })
            df = pd.DataFrame(transaction_data)
            
            # Initialize predictor
            predictor = StockPredictor()
            
            # Train model
            try:
                predictor.prepare_data(df, product_id=selected_product_id)
                
                # Default SARIMA parameters (can be tuned)
                params = {
                    'p': 1, 'd': 1, 'q': 1,  # Non-seasonal components
                    'P': 1, 'D': 1, 'Q': 1, 's': 7  # Seasonal components (weekly)
                }
                
                model = predictor.train(params=params)
                
                # Make predictions
                forecast_result = predictor.predict(steps=prediction_days)
                forecast = forecast_result['forecast']
                confidence_intervals = forecast_result['confidence_intervals']
                
                # Calculate reorder point
                reorder_info = predictor.calculate_reorder_point(lead_time)
                
                # Generate visualization
                plt.figure(figsize=(10, 6))
                plt.plot(predictor.history_data[-30:].index, predictor.history_data[-30:], label='Historical')
                plt.plot(forecast.index, forecast, color='red', label='Forecast')
                plt.fill_between(
                    confidence_intervals.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1], 
                    color='pink', alpha=0.3,
                    label='95% Confidence Interval'
                )
                
                # Add horizontal line for reorder point
                plt.axhline(y=reorder_info['reorder_point'], color='green', linestyle='--', 
                           label=f'Reorder Point ({reorder_info["reorder_point"]:.2f})')
                
                plt.title(f'Prediksi Stok: {selected_product.nama}')
                plt.xlabel('Tanggal')
                plt.ylabel('Jumlah')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save plot to buffer
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
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
                
                context.update({
                    'plot_data': plot_data,
                    'forecast_dates': json.dumps(forecast_dates),
                    'forecast_values': json.dumps(forecast_values),
                    'reorder_info': reorder_info,
                    'current_stock': selected_product.stok,
                    'days_until_stockout': calculate_days_until_stockout(forecast, selected_product.stok)
                })
                
            except Exception as e:
                messages.error(request, f"Error during prediction: {str(e)}")
                import traceback
                print(traceback.format_exc())
        
        except Exception as e:
            messages.error(request, f"Error: {str(e)}")
    
    return render(request, 'page/stock_prediction.html', context)

def calculate_days_until_stockout(forecast, current_stock):
    """Calculate how many days until stock is depleted based on forecast"""
    if current_stock <= 0:
        return 0
        
    cumulative_demand = 0
    for i, value in enumerate(forecast.values):
        cumulative_demand += value
        if cumulative_demand >= current_stock:
            return i + 1
            
    return len(forecast)  # Stock will last beyond forecast period

@login_required
def get_product_history(request, product_id):
    """AJAX endpoint to get product transaction history"""
    try:
        product = get_object_or_404(Stok, id=product_id)
        transactions = Transaksi.objects.filter(
            stok_id=product_id
        ).order_by('-tanggal_transaksi')[:30]  # Last 30 transactions
        
        data = []
        for t in transactions:
            data.append({
                'date': t.tanggal_transaksi.strftime('%Y-%m-%d'),
                'qty': t.qty,
                'total': t.total_harga
            })
            
        return JsonResponse({'status': 'success', 'data': data})
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)})