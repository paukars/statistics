from django.urls import path

from .views import (histograma_de_benignos, histograma_de_malignos,
                    histogramas_combinados, malignos_curva, benignos_curva)

urlpatterns = [
    path("histograma-maligno", histograma_de_malignos, name="histograma-maligno"),
    path("histograma-benigno", histograma_de_benignos, name="histograma-benigno"),
    path("histograma-combinado", histogramas_combinados, name="histogramas_combinados"),
    path("malignos-curva", malignos_curva, name="malignos_curva"),
    path("benignos-curva", benignos_curva, name="benignos_curva"),
]
