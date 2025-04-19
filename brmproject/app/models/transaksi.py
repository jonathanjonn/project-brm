from django.db import models

class Transaksi(models.Model):
    id = models.AutoField(primary_key=True)
    stok_id = models.ForeignKey('Stok', on_delete=models.SET_NULL, null=True)
    qty = models.IntegerField()
    total_harga = models.IntegerField()
    tanggal_transaksi = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)