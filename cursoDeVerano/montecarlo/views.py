import math

import numpy as np
from django.http import JsonResponse
from django.shortcuts import render

# Create your views here.


def montecarlo_integral(request, num_samples):
    num_samples = int(num_samples)

    def integrand(x):
        return math.cos(x**2) * math.sin(x**4)

    random_samples = np.random.uniform(0, 1, num_samples)

    sample_values = np.array([integrand(x) for x in random_samples])

    mean_value = np.mean(sample_values)

    integral_estimate = mean_value

    return JsonResponse(
        {"num_samples": num_samples, "integral_estimate": integral_estimate}
    )


def montecarlo_exponencial(request, num_samples):
    num_samples = int(num_samples)

    def integrand(x):
        return 25 * x**2 * np.cos(x**2) * np.exp(-25 * x)

    samples = np.random.exponential(scale=1.0 / 25, size=num_samples)

    evaluations = integrand(samples)

    integral_estimate = np.mean(evaluations)

    return JsonResponse(
        {"num_samples": num_samples, "integral_estimate": integral_estimate}
    )
