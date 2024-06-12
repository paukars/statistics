from django.contrib import admin

from .models import ReporteMedico

# Register your models here.


class ReporteMedicoAdmin(admin.ModelAdmin):
    list_display = ("Id", "Diagnosis", "Textura", "Perimetro")
    list_filter = ("Diagnosis",)
    search_fields = ("Id", "Diagnosis")
admin.site.register(ReporteMedico, ReporteMedicoAdmin)
