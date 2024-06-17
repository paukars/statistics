from django.urls import path

from .views import montecarlo_exponencial, montecarlo_integral

urlpatterns = [
    path(
        "integral-montecarlo/<int:num_samples>",
        montecarlo_integral,
        name="montecarlo-integral",
    ),
    path(
        "montencarlo-exponencial/<int:num_samples>",
        montecarlo_exponencial,
        name="montencarlo-exponencial",
    ),
]
