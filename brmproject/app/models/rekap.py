from django.db import models
from django.utils import timezone

class RekapBulanan(models.Model):
    stok = models.ForeignKey('Stok', on_delete=models.CASCADE)
    bulan = models.DateField() 

    stok_awal = models.IntegerField()
    total_masuk = models.IntegerField()
    total_keluar = models.IntegerField()
    stok_akhir = models.IntegerField()

    rekomendasi_pembelian = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('stok', 'bulan')

    def __str__(self):
        return f"{self.stok.nama} - {self.bulan.strftime('%B %Y')}"
