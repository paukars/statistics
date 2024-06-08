from django.urls import path

from .views import histograma_de_malignos, histograma_de_benignos, histogramas_combinados

urlpatterns = [
    path("histograma-maligno", histograma_de_malignos, name="histograma-maligno"),
    path("histograma-benigno", histograma_de_benignos, name="histograma-benigno"),
    path("histograma-combinado", histogramas_combinados, name="histogramas_combinados")
]
