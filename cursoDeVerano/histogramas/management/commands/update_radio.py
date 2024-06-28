import csv

from django.core.management.base import BaseCommand
from django.db import transaction
from histogramas.models import ReporteMedico


class Command(BaseCommand):
    help = "Update Radio field from CSV file"

    def handle(self, *args, **kwargs):
        csv_file_path = "/home/paukars/Downloads/base.csv"  # Path to your CSV file

        with open(csv_file_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            with transaction.atomic():
                for row in reader:
                    try:
                        reporte = ReporteMedico.objects.get(
                            Id=row["Id"]
                        )  # Adjust 'ID' to match the column name in your CSV
                        reporte.Radio = float(
                            row["radius"]
                        )  # Adjust 'radius' to match the column name in your CSV
                        reporte.save()
                    except ReporteMedico.DoesNotExist:
                        self.stdout.write(
                            self.style.WARNING(
                                f"ReporteMedico with ID {row['Id']} does not exist."
                            )
                        )
                    except ValueError as e:
                        self.stdout.write(
                            self.style.ERROR(f"Error processing  {row['Id']}: {e}")
                        )
