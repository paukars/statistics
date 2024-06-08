from django.db import models

# Create your models here.

class ReporteMedico(models.Model):
    Id = models.IntegerField(primary_key=True)
    Diagnosis = models.CharField(max_length=1)
    Texture = models.FloatField()
    Perimeter = models.FloatField()



