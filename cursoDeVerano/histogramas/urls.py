from django.urls import path

from .views import histograma_de_malignos

urlpatterns = [
    path("histograma-maligno", histograma_de_malignos, name="histograma-maligno")
]
