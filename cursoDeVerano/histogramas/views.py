import io
import numpy as np
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render

from .models import ReporteMedico

# Create your views here.


def histograma_de_malignos(request):
    records = ReporteMedico.objects.filter(Diagnosis="M")

    perimeters = [record.Perimeter for record in records]

    plt.figure(figsize=(10, 9))
    # plt.hist(perimeters, bins=bins, color="blue", edgecolor="black", density=True)
    counts, bins, patches = plt.hist(
        perimeters, bins=30, density=True, color="red", edgecolor="black"
    )
    plt.title("Histograma del perimetro para diagnosticos malignos")
    plt.xlabel("Perimetro")
    plt.ylabel("Frecuencia")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(bins, rotation="vertical")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    response = HttpResponse(buf.getvalue(), content_type="image/png")
    response["Content-Length"] = buf.tell()

    buf.seek(0)
    return response


def histograma_de_benignos(request):
    records = ReporteMedico.objects.filter(Diagnosis="B")

    perimeters = [record.Perimeter for record in records]

    plt.figure(figsize=(10, 9))
    # plt.hist(perimeters, bins=bins, color="blue", edgecolor="black", density=True)
    counts, bins, patches = plt.hist(
        perimeters, bins=40, density=True, color="green", edgecolor="black"
    )
    plt.title("Histograma del perimetro para diagnosticos malignos")
    plt.xlabel("Perimetro")
    plt.ylabel("Frecuencia")
    plt.grid(True, which="both", axis="x", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(True, which="both", axis="y", color="gray", linestyle="-", linewidth=0.5)
    plt.xticks(bins, rotation="vertical")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    response = HttpResponse(buf.getvalue(), content_type="image/png")
    response["Content-Length"] = buf.tell()

    buf.seek(0)
    return response


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
        "mean_perimeter_malignant" : mean_perimeter_malignant,
        "std_perimeter_malignant": std_perimeter_malignant,
        "mean_perimeter_benign": mean_perimeter_benign,
        "std_perimeter_benign": std_perimeter_benign,
    }

    return render(request, "histogramas_combinados.html", context)
