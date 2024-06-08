from django.contrib import admin

from .models import ReporteMedico

# Register your models here.


class ReporteMedicoAdmin(admin.ModelAdmin):
    list_display = ("Id", "Diagnosis", "Texture", "Perimeter")
    list_filter = ("Diagnosis",)
    search_fields = ("Id", "Diagnosis")
admin.site.register(ReporteMedico, ReporteMedicoAdmin)
