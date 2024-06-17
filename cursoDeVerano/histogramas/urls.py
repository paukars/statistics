from django.urls import path

from .views import CombinedHistogramView, histogramas_combinados

urlpatterns = [
    path("histograma-combinado", histogramas_combinados, name="histogramas_combinados"),
    path("<int:bins>/<int:bins3d>/", CombinedHistogramView.as_view()),
    path(
        "<int:bins>/",
        CombinedHistogramView.as_view(),
        name="combined_histograms",
    ),
    path(
        "",
        CombinedHistogramView.as_view(),
        name="combined_histograms_default",
    ),
]
