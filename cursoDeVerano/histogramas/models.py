from django.db import models

# Create your models here.

class ReporteMedico(models.Model):
    Id = models.IntegerField(primary_key=True)
    Diagnosis = models.CharField(max_length=1)
    Textura = models.FloatField()
    Perimetro = models.FloatField()



