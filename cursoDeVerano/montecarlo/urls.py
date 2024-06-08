from django.urls import path

from .views import montecarlo_integral

urlpatterns = [
    path(
        "integral-montecarlo/<int:num_samples>",
        montecarlo_integral,
        name="montecarlo-integral",
    ),
]
