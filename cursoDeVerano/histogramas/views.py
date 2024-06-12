import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from django.views import View
from django.http import HttpResponse
from django.shortcuts import render
from scipy import stats

from .models import ReporteMedico

# Create your views here.


class HistogramBaseView(View):
    diagnosis_type = None
    color = None
    bins = None

    def get_queryset(self):
        return ReporteMedico.objects.filter(Diagnosis=self.diagnosis_type)

    def get_histogram_data(self):
        records = self.get_queryset()
        perimeters = [record.Perimeter for record in records]
        return perimeters

    def generate_histogram(self, perimeters):
        plt.figure(figsize=(10, 9))
        counts, bins, patches = plt.hist(
            perimeters, bins=self.bins, density=True, color=self.color, edgecolor="black"
        )
        plt.title(f"Histograma del perimetro para diagnosticos {self.diagnosis_type}")
        plt.xlabel("Perimetro")
        plt.ylabel("Frecuencia")
        plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
        plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
        plt.xticks(bins, rotation="vertical")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return image_base64

class CombinedHistogramView(HistogramBaseView):
    def get(self, request, *args, **kwargs):
        # Generate histogram for malignant diagnoses
        self.diagnosis_type = "M"
        self.color = "red"
        self.bins = 30
        malignant_data = self.get_histogram_data()
        malignant_image = self.generate_histogram(malignant_data)

        # Generate histogram for benign diagnoses
        self.diagnosis_type = "B"
        self.color = "green"
        self.bins = 40
        benign_data = self.get_histogram_data()
        benign_image = self.generate_histogram(benign_data)

        # Render the template with both images
        context = {
            'malignant_image': malignant_image,
            'benign_image': benign_image,
        }
        return render(request, 'combined_histograms.html', context)


def histogramas_combinados(request):
    malignant_records = ReporteMedico.objects.filter(Diagnosis="M")
    benign_records = ReporteMedico.objects.filter(Diagnosis="B")

    malignant_perimeters = [record.Perimeter for record in malignant_records]
    benign_perimeters = [record.Perimeter for record in benign_records]

    # Calculate mean and standard deviation for each diagnosis
    mean_perimeter_malignant = np.mean(malignant_perimeters)
    std_perimeter_malignant = np.std(malignant_perimeters)
    mean_perimeter_benign = np.mean(benign_perimeters)
    std_perimeter_benign = np.std(benign_perimeters)

    # Create the histogram plot
    plt.figure(figsize=(12, 9))

    # Plot the combined histogram
    counts, bins, patches = plt.hist(
        [malignant_perimeters, benign_perimeters],
        bins=30,
        density=True,
        color=["red", "green"],
        edgecolor="black",
        label=["Malignos", "Benignos"],
        alpha=0.7,
        align="left",
    )

    plt.xticks(bins, rotation="vertical")

    # Plot title and labels
    plt.title("Histograma del Perímetro para Diagnósticos Malignos y Benignos")
    plt.xlabel("Perímetro")
    plt.ylabel("Frecuencia")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)

    # Save the figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image_data = buf.getvalue()

    context = {
        "image_data": image_data,
        "mean_perimeter_malignant": mean_perimeter_malignant,
        "std_perimeter_malignant": std_perimeter_malignant,
        "mean_perimeter_benign": mean_perimeter_benign,
        "std_perimeter_benign": std_perimeter_benign,
    }

    return render(request, "histogramas_combinados.html", context)


def malignos_curva(request):
    records = ReporteMedico.objects.filter(Diagnosis="M")
    perimeters = [record.Perimeter for record in records]

    # Calculate mean and standard deviation for malignant diagnoses
    mean_perimeter = np.mean(perimeters)
    std_perimeter = np.std(perimeters)

    # Fit different distributions and determine the best fit
    distributions = [stats.expon, stats.lognorm, stats.chi2, stats.norm]
    best_fit = None
    best_ks_stat = np.inf

    for distribution in distributions:
        params = distribution.fit(perimeters)
        ks_stat, p_value = stats.kstest(perimeters, distribution.cdf, args=params)
        if ks_stat < best_ks_stat:
            best_ks_stat = ks_stat
            best_fit = distribution, params

    # Create the histogram plot
    plt.figure(figsize=(10, 9))
    counts, bins, patches = plt.hist(
        perimeters, bins=30, density=True, color="red", edgecolor="black"
    )
    plt.title("Histograma del perimetro para diagnosticos malignos")
    plt.xlabel("Perimetro")
    plt.ylabel("Frecuencia")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(bins, rotation="vertical")

    # Save the histogram figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)
    histogram_image_data = buf.getvalue()

    # Q-Q plot for malignant perimeters
    distribution, params = best_fit
    plt.figure(figsize=(10, 9))
    stats.probplot(perimeters, dist=distribution, sparams=params, plot=plt)
    plt.title("Q-Q Plot para Diagnósticos malignos")
    plt.xlabel(f"Cuantiles teoricos la Distribución {best_fit[0].name.capitalize()}")
    plt.ylabel("Datos ordenados")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    qq_buf = io.BytesIO()
    plt.savefig(qq_buf, format="png")
    plt.close()
    qq_buf.seek(0)
    qq_image_data = qq_buf.getvalue()

    context = {
        "histogram_image_data": histogram_image_data,
        "qq_image_data": qq_image_data,
        "mean_perimeter": mean_perimeter,
        "std_perimeter": std_perimeter,
        "best_fit_distribution": best_fit[0].name,
    }

    return render(request, "malignos_curva.html", context)


def benignos_curva(request):
    records = ReporteMedico.objects.filter(Diagnosis="B")

    perimeters = [record.Perimeter for record in records]

    # Calculate mean and standard deviation for benign diagnoses
    mean_perimeter = np.mean(perimeters)
    std_perimeter = np.std(perimeters)

    # Fit different distributions and determine the best fit
    distributions = [stats.norm, stats.expon, stats.lognorm, stats.chi2]
    best_fit = None
    best_ks_stat = np.inf

    for distribution in distributions:
        params = distribution.fit(perimeters)
        ks_stat, p_value = stats.kstest(perimeters, distribution.cdf, args=params)
        if ks_stat < best_ks_stat:
            best_ks_stat = ks_stat
            best_fit = distribution, params

    # Create the histogram plot
    plt.figure(figsize=(10, 9))
    counts, bins, patches = plt.hist(
        perimeters, bins=40, density=True, color="green", edgecolor="black"
    )
    plt.title("Histograma del perimetro para diagnosticos benignos")
    plt.xlabel("Perimetro")
    plt.ylabel("Frecuencia")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(bins, rotation="vertical")

    # Save the histogram figure to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    buf.seek(0)
    histogram_image_data = buf.getvalue()

    # Q-Q plot for benign perimeters
    distribution, params = best_fit
    plt.figure(figsize=(10, 9))
    stats.probplot(perimeters, dist=distribution, sparams=params, plot=plt)
    plt.title("Q-Q Plot para Diagnósticos Benignos")
    plt.xlabel(f"Cuantiles teoricos de la Distribución {best_fit[0].name.capitalize()}")
    plt.ylabel("Datos ordenados")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    qq_buf = io.BytesIO()
    plt.savefig(qq_buf, format="png")
    plt.close()
    qq_buf.seek(0)
    qq_image_data = qq_buf.getvalue()

    context = {
        "histogram_image_data": histogram_image_data,
        "qq_image_data": qq_image_data,
        "mean_perimeter": mean_perimeter,
        "std_perimeter": std_perimeter,
        "best_fit_distribution": best_fit[0].name,
    }

    return render(request, "benignos_curva.html", context)
