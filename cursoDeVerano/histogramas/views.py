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
from sklearn.mixture import GaussianMixture

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

    def generate_combined_histogram(
        self, data1, data2, label1, label2, title, color1, color2, bins
    ):
        plt.figure(figsize=(12, 10))

        # Plot histograms
        plt.hist(
            data1,
            bins=bins,
            density=True,
            color=color1,
            edgecolor="black",
            alpha=0.5,
            label=label1,
        )
        plt.hist(
            data2,
            bins=bins,
            density=True,
            color=color2,
            edgecolor="black",
            alpha=0.5,
            label=label2,
        )

        # Plot normal distribution curves
        mu1, std1 = np.mean(data1), np.std(data1)
        mu2, std2 = np.mean(data2), np.std(data2)
        median1, median2 = np.median(data1), np.median(data2)
        cov1, cov2 = np.var(data1), np.var(
            data2
        )  # Using variance instead of covariance matrix for 1D data
        x = np.linspace(plt.xlim()[0], plt.xlim()[1], 100)

        p1 = norm.pdf(x, mu1, std1)
        p2 = norm.pdf(x, mu2, std2)

        plt.plot(x, p1, color=color1, linewidth=1.5)
        plt.plot(x, p2, color=color2, linewidth=1.5)

        # Add text annotations for variance and median
        textstr1 = "\n".join(
            (f"{label1}", f"$\\mu={mu1:.2f}$", f"$\\sigma={std1:.2f}$")
        )
        plt.gcf().text(
            0.15, 0.8, textstr1, fontsize=12, bbox=dict(facecolor="white", alpha=0.5)
        )

        textstr2 = "\n".join(
            (f"{label2}", f"$\\mu={mu2:.2f}$", f"$\\sigma={std2:.2f}$")
        )
        plt.gcf().text(
            0.15, 0.7, textstr2, fontsize=12, bbox=dict(facecolor="white", alpha=0.5)
        )

        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend(loc="upper right")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_gmm_plot(self, data1, data2, title, color1, color2, bins):
        combined_data = np.concatenate([data1, data2])
        combined_data = combined_data.reshape(-1, 1)

        """gmm = GaussianMixture(
            n_components=2, n_init=2000, init_params="random_from_data"
        )"""
        gmm = GaussianMixture(n_components=2)
        gmm.fit(combined_data)
        means = gmm.means_.flatten()
        covariances = gmm.covariances_.flatten()
        weights = gmm.weights_

        x = np.linspace(min(combined_data), max(combined_data), 1000).reshape(-1, 1)
        pdfs = [
            w * norm.pdf(x, m, np.sqrt(c))
            for w, m, c in zip(weights, means, covariances)
        ]
        pdf = np.sum(pdfs, axis=0)

        plt.figure(figsize=(12, 10))
        plt.hist(
            combined_data,
            bins=bins,
            density=True,
            color="grey",
            edgecolor="black",
            alpha=0.5,
            label="Datos completos",
        )
        plt.plot(x, pdf, "k-", label="GMM")

        for i, (mean, cov, w) in enumerate(zip(means, covariances, weights)):
            std_dev = np.sqrt(cov)
            plt.plot(x, w * norm.pdf(x, mean, std_dev), label=f"Componente {i+1}")
            textstr = "\n".join(
                (
                    f"Componente {i+1}",
                    f"$\mu={mean:.2f}$",
                    f"$\sigma={std_dev:.2f}$",
                    f"$\pi={w:.2f}$",
                )
            )
            plt.gcf().text(
                0.75,
                0.5 - i * 0.1,
                textstr,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        plt.title(title)
        plt.xlabel("Valores")
        plt.ylabel("Densidad")
        plt.legend(loc="upper right")
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_3d_histogram(
        self,
        perimeter_values,
        texture_values,
        title,
        bins=10,
        x_range=None,
        y_range=None,
    ):
        x = np.array(perimeter_values)
        y = np.array(texture_values)

        # Set default ranges if none provided
        if x_range is None:
            x_range = [x.min(), x.max()]
        if y_range is None:
            y_range = [y.min(), y.max()]

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(projection="3d")

        # hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[x_range, y_range])

        # Create the 3D histogram
        hist, xedges, yedges = np.histogram2d(
            x, y, bins=(bins, bins), range=[x_range, y_range], density=True
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

        # Plot bivariate Gaussian distribution
        mean = [np.mean(x), np.mean(y)]
        cov = np.cov(x, y)

        x_mesh, y_mesh = np.meshgrid(
            np.linspace(x_range[0], x_range[1], 100),
            np.linspace(y_range[0], y_range[1], 100),
        )
        pos = np.dstack((x_mesh, y_mesh))
        rv = multivariate_normal(mean, cov)
        z_mesh = rv.pdf(pos)

        # ax.plot_wireframe(x_mesh, y_mesh, z_mesh, color="green")
        ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap="viridis", alpha=0.6)

        ax.set_title(f"Grafico en 3D para perimetro y textura - {title}")
        ax.set_xlabel("Perimetro")
        ax.set_ylabel("Textura")

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
        combined_histograms = []
        gmm_plots = []

        # Existing attributes
        attributes = [
            ("Perimetro", "M", "red"),
            ("Perimetro", "B", "green"),
            ("Textura", "M", "blue"),
            ("Textura", "B", "orange"),
            ("Radio", "M", "purple"),
            ("Radio", "B", "yellow"),
        ]

        # Generate histograms for existing attributes
        for attribute, diagnosis, color in attributes:
            self.attribute = attribute
            self.diagnosis_type = diagnosis
            self.color = color
            data = self.get_histogram_data(attribute)
            if data:  # Ensure data is not empty
                image = self.generate_histogram(data, diagnosis, attribute, bins)
                histograms.append(
                    {
                        "attribute": attribute,
                        "diagnosis": diagnosis,
                        "image": image,
                    }
                )

        # Generate combined histograms for each attribute
        combined_attributes = [
            ("Perimetro", "red", "green"),
            ("Textura", "blue", "orange"),
            ("Radio", "purple", "yellow"),
        ]
        for attribute, color_m, color_b in combined_attributes:
            self.attribute = attribute

            # Get data for both diagnoses
            self.diagnosis_type = "M"
            data_m = self.get_histogram_data(attribute)
            self.diagnosis_type = "B"
            data_b = self.get_histogram_data(attribute)

            if data_m and data_b:  # Ensure data is not empty
                combined_image = self.generate_combined_histogram(
                    data_m,
                    data_b,
                    "Malignos",
                    "Benignos",
                    f"Histograma de {attribute} para Diagnósticos Malignos y Benignos",
                    color_m,
                    color_b,
                    bins,
                )
                combined_histograms.append(
                    {
                        "attribute": attribute,
                        "image": combined_image,
                    }
                )
                gmm_image = self.generate_gmm_plot(
                    data_m,
                    data_b,
                    f"GMM de {attribute} para Diagnósticos Malignos y Benignos",
                    color_m,
                    color_b,
                    bins,
                )
                gmm_plots.append(
                    {
                        "attribute": attribute,
                        "image": gmm_image,
                    }
                )

        # Generate 3D histograms for Perimetro and Textura
        self.diagnosis_type = "B"
        perimeter_values_b = self.get_histogram_data("Perimetro")
        texture_values_b = self.get_histogram_data("Textura")
        plot_3d_histogram_b = None
        if perimeter_values_b and texture_values_b:  # Ensure data is not empty
            plot_3d_histogram_b = self.generate_3d_histogram(
                perimeter_values_b, texture_values_b, "Benignos", bins3d
            )

        self.diagnosis_type = "M"
        perimeter_values_m = self.get_histogram_data("Perimetro")
        texture_values_m = self.get_histogram_data("Textura")
        plot_3d_histogram_m = None
        if perimeter_values_m and texture_values_m:  # Ensure data is not empty
            plot_3d_histogram_m = self.generate_3d_histogram(
                perimeter_values_m,
                texture_values_m,
                "Malignos",
                bins3d,
            )

        plot_3d_histogram_combined = None
        if (
            perimeter_values_b
            and perimeter_values_m
            and texture_values_b
            and texture_values_m
        ):  # Ensure data is not empty
            plot_3d_histogram_combined = self.generate_3d_histogram(
                perimeter_values_b + perimeter_values_m,
                texture_values_b + texture_values_m,
                "Combinados",
                bins3d,
            )

        # Calculate the indices for the histograms
        histogram_length = len(histograms)
        combined_histogram_length = len(combined_histograms)
        gmm_plots_length = len(gmm_plots)
        index_3d_b = histogram_length + combined_histogram_length + gmm_plots_length
        index_3d_m = histogram_length + combined_histogram_length + gmm_plots_length + 1
        index_3d_combined = (
            histogram_length + combined_histogram_length + gmm_plots_length + 2
        )

        # Precompute indices for each plot type
        combined_histogram_indices = [
            histogram_length + i for i in range(combined_histogram_length)
        ]
        gmm_plot_indices = [
            histogram_length + combined_histogram_length + i
            for i in range(gmm_plots_length)
        ]

        context = {
            "histograms": histograms,
            "combined_histograms": combined_histograms,
            "gmm_plots": gmm_plots,
            "plot_3d_histogram_b": plot_3d_histogram_b,
            "plot_3d_histogram_m": plot_3d_histogram_m,
            "plot_3d_histogram_combined": plot_3d_histogram_combined,
            "index_3d_b": index_3d_b,
            "index_3d_m": index_3d_m,
            "index_3d_combined": index_3d_combined,
            "combined_histogram_indices": combined_histogram_indices,
            "gmm_plot_indices": gmm_plot_indices,
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
