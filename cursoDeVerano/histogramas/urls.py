from django.urls import path

from .views import  histogramas_combinados, CombinedHistogramView

urlpatterns = [
    path("histograma-combinado", histogramas_combinados, name="histogramas_combinados"),
    path('histograms/', CombinedHistogramView.as_view(), name='combined_histograms'),
]
