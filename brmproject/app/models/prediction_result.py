from django.db import models
from django.utils import timezone
import json

class StockPredictionResult(models.Model):
    """
    Model untuk menyimpan hasil prediksi stok
    
    Menyimpan hasil prediksi untuk menghindari komputasi berulang dan
    untuk referensi historis.
    """
    product = models.ForeignKey('Stok', on_delete=models.CASCADE, related_name='predictions')
    prediction_date = models.DateTimeField(default=timezone.now)
    current_stock = models.FloatField(default=0)
    forecast_data = models.TextField()  
    reorder_point = models.FloatField(default=0)
    expected_demand = models.FloatField(default=0)
    days_until_stockout = models.IntegerField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-prediction_date']
        indexes = [
            models.Index(fields=['product', 'prediction_date']),
        ]
    
    def __str__(self):
        return f"Prediksi untuk {self.product.nama} pada {self.prediction_date.strftime('%Y-%m-%d')}"
    
    def get_forecast(self):
        """
        Mendapatkan data forecast dalam format yang sudah di-parse
        """
        try:
            return json.loads(self.forecast_data)
        except (ValueError, TypeError):
            return {'forecast': [], 'dates': []}
    
    @property
    def status(self):
        """
        Mendapatkan status stok berdasarkan reorder point
        """
        if self.current_stock <= 0:
            return 'danger'  # Stok habis
        elif self.current_stock <= self.reorder_point:
            return 'warning'  # Perlu reorder
        else:
            return 'success'  # Stok aman
    
    @property
    def status_text(self):
        """
        Mendapatkan teks status stok
        """
        status_map = {
            'danger': 'Stok Habis',
            'warning': 'Perlu Reorder',
            'success': 'Stok Aman'
        }
        return status_map.get(self.status, 'Tidak Diketahui')
