import io

import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render

from .models import ReporteMedico

# Create your views here.


def histograma_de_malignos(request):
    records = ReporteMedico.objects.filter(Diagnosis="M")

    perimeters = [record.Perimeter for record in records]

    
    plt.figure(figsize=(10, 9))
    #plt.hist(perimeters, bins=bins, color="blue", edgecolor="black", density=True)
    counts, bins, patches = plt.hist(perimeters, bins=40, density=True, color='blue', edgecolor='black')
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
