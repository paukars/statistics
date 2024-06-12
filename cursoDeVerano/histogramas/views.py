import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from scipy.stats import norm

from .models import ReporteMedico

# Create your views here.


class HistogramBaseView(View):
    diagnosis_type = None
    color = None
    bins = None
    attribute = None
    title_map = {"M": "malignos", "B": "benignos"}

    def get_queryset(self):
        return ReporteMedico.objects.filter(Diagnosis=self.diagnosis_type)

    def get_histogram_data(self, attribute):
        records = self.get_queryset()
        values = [getattr(record, attribute) for record in records]
        return values

    def generate_histogram(self, values, diagnosis, attribute):
        plt.figure(figsize=(10, 9))

        counts, bins, patches = plt.hist(
            values,
            bins=self.bins,
            density=True,
            color=self.color,
            edgecolor="black",
        )

        # Plot normal distribution curve
        mu, std = norm.fit(values)  # Calculate mean and standard deviation
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, "k", linewidth=1.5)

        diagnosis_title = "malignos" if diagnosis == "M" else "benignos"
        plt.title(f"Histograma del {attribute} para diagnosticos {diagnosis_title}")
        plt.xlabel(attribute.capitalize())
        plt.ylabel("Frecuencia")
        plt.grid(
            True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5
        )
        plt.grid(
            True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5
        )
        plt.xticks(bins, rotation="vertical")

        # Annotate mean and standard deviation
        textstr = "\n".join((f"$\mu={mu:.2f}$", f"$\sigma={std:.2f}$"))
        plt.gcf().text(
            0.75, 0.75, textstr, fontsize=12, bbox=dict(facecolor="white", alpha=0.5)
        )


        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image_base64


class CombinedHistogramView(HistogramBaseView):
    def get(self, request, *args, **kwargs):
        histograms = []
        attributes = [
            ("Perimetro", 30, "M", "red"),
            ("Perimetro", 40, "B", "green"),
            ("Textura", 30, "M", "blue"),
            ("Textura", 40, "B", "orange"),
        ]
        for attribute, bins, diagnosis, color in attributes:
            self.attribute = attribute
            self.bins = bins
            self.diagnosis_type = diagnosis
            self.color = color
            data = self.get_histogram_data(attribute)
            image = self.generate_histogram(data, diagnosis, attribute)
            histograms.append(
                {
                    "attribute": attribute,
                    "diagnosis": diagnosis,
                    "image": image,
                }
            )

        context = {
            "histograms": histograms,
        }
        return render(request, "combined_histograms.html", context)


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
