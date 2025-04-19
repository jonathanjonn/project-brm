from django.db import models

class Transaksi(models.Model):
    TIPE_TRANSAKSI = [
        (1, 'Masuk (Pembelian dari pusat)'),
        (2, 'Keluar (Penjualan)'),
    ]
    id = models.AutoField(primary_key=True)
    stok_id = models.ForeignKey('Stok', on_delete=models.SET_NULL, null=True)
    qty = models.IntegerField()
    total_harga = models.IntegerField()
    tanggal_transaksi = models.DateField()
    tipe = models.IntegerField(choices=TIPE_TRANSAKSI)
    created_at = models.DateTimeField(auto_now_add=True)