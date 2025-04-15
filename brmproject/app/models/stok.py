from django.db import models

class Stok(models.Model):
    id = models.AutoField(primary_key=True)
    nama = models.CharField(max_length=100)
    kategori = models.CharField(max_length=100)
    harga = models.IntegerField()
    stok = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.nama