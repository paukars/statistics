import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.stats import multivariate_normal, norm

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

    def generate_histogram(self, values, diagnosis, attribute, bins):
        plt.figure(figsize=(10, 9))

        counts, bins, patches = plt.hist(
            values,
            bins=bins,
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
        plt.title(f"Histograma de {attribute} para diagnosticos {diagnosis_title}")
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

    def generate_3d_histogram(
        self, perimeter_values, texture_values, bins=10, x_range=None, y_range=None
    ):
        x = np.array(perimeter_values)
        y = np.array(texture_values)

        # Set default ranges if none provided
        if x_range is None:
            x_range = [x.min(), x.max()]
        if y_range is None:
            y_range = [y.min(), y.max()]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        # hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])

        # Create the 3D histogram
        hist, xedges, yedges = np.histogram2d(
            x, y, bins=(bins, bins), range=[x_range, y_range]
        )
        xpos, ypos = np.meshgrid(
            xedges[:-1] + (xedges[1] - xedges[0]) / 2,
            yedges[:-1] + (yedges[1] - yedges[0]) / 2,
            indexing="ij",
        )
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)

        dx = (xedges[1] - xedges[0]) * np.ones_like(zpos)
        dy = (yedges[1] - yedges[0]) * np.ones_like(zpos)
        dz = hist.ravel()

        # Get desired colormap
        cmap = plt.get_cmap("jet")
        max_height = np.max(dz)  # get range of colorbars so we can normalize
        min_height = np.min(dz)
        rgba = [
            cmap((k - min_height) / max_height) for k in dz
        ]  # scale each z to [0,1], and get their rgb values

        # Plot the bars
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=rgba, zsort="average")

        # Fit a bivariate normal distribution to the data
        mu = [np.mean(x), np.mean(y)]
        sigma = np.cov(x, y)
        rv = multivariate_normal(mu, sigma)

        # Create a grid for plotting the normal distribution
        x_grid = np.linspace(x_range[0], x_range[1], 100)
        y_grid = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        pos = np.dstack((X, Y))
        Z = rv.pdf(pos)

        # Plot the normal distribution surface

        # Plot the normal distribution surface
        ax.plot_surface(X, Y, Z, cmap="viridis", alpha=1)

        ax.set_title("3D Plot of Perimetro and Textura")
        ax.set_xlabel("Perimetro")
        ax.set_ylabel("Textura")
        ax.set_zlabel("Density")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return image_base64


class CombinedHistogramView(HistogramBaseView):
    def get(self, request, bins=20, bins3d=10):
        bins = int(bins)

        histograms = []

        attributes = [
            ("Perimetro", "M", "red"),
            ("Perimetro", "B", "green"),
            ("Textura", "M", "blue"),
            ("Textura", "B", "orange"),
        ]
        for attribute, diagnosis, color in attributes:
            self.attribute = attribute
            self.diagnosis_type = diagnosis
            self.color = color
            data = self.get_histogram_data(attribute)
            image = self.generate_histogram(data, diagnosis, attribute, bins)
            histograms.append(
                {
                    "attribute": attribute,
                    "diagnosis": diagnosis,
                    "image": image,
                }
            )

        # Generate 3D histogram for Perimetro and Textura

        # Generate 3D histograms for Perimetro and Textura
        self.diagnosis_type = "B"
        perimeter_values_b = self.get_histogram_data("Perimetro")
        texture_values_b = self.get_histogram_data("Textura")
        plot_3d_histogram_b = self.generate_3d_histogram(
            perimeter_values_b, texture_values_b, bins3d
        )

        self.diagnosis_type = "M"
        perimeter_values_m = self.get_histogram_data("Perimetro")
        texture_values_m = self.get_histogram_data("Textura")
        plot_3d_histogram_m = self.generate_3d_histogram(
            perimeter_values_m, texture_values_m, bins3d
        )

        plot_3d_histogram_combined = self.generate_3d_histogram(
            perimeter_values_b + perimeter_values_m,
            texture_values_b + texture_values_m,
            bins3d,
        )

        # Calculate the indices for the 3D histograms
        histogram_length = len(histograms)
        index_3d_b = histogram_length
        index_3d_m = histogram_length + 1
        index_3d_combined = histogram_length + 2

        context = {
            "histograms": histograms,
            "plot_3d_histogram_b": plot_3d_histogram_b,
            "plot_3d_histogram_m": plot_3d_histogram_m,
            "plot_3d_histogram_combined": plot_3d_histogram_combined,
            "index_3d_b": index_3d_b,
            "index_3d_m": index_3d_m,
            "index_3d_combined": index_3d_combined,
        }

        return render(request, "combined_histograms.html", context)


def histogramas_combinados(request):
    malignant_records = ReporteMedico.objects.filter(Diagnosis="M")
    benign_records = ReporteMedico.objects.filter(Diagnosis="B")

    malignant_perimeters = [record.Perimetro for record in malignant_records]
    benign_perimeters = [record.Perimetro for record in benign_records]

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
