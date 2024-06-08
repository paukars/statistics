import openpyxl
from django.core.management.base import BaseCommand
from openpyxl.workbook import workbook

from histogramas.models import ReporteMedico


class Command(BaseCommand):
    help = "Uploads data from an XLSX file to the ReportMedico model"

    def add_arguments(self, parser):
        parser.add_argument("filepath", type=str, help="The path to the XLSX file")

    def handle(self, *args, **options):
        filepath = options["filepath"]
        self.stdout.write(f"Loading data from {filepath}")
        workbook = openpyxl.load_workbook(filepath)
        sheet = workbook.active
        for row in sheet.iter_rows(min_row=2, values_only=True):
            ReporteMedico.objects.update_or_create(
                id=row[0],
                defaults={"diagnosis": row[1], "texture": row[2], "perimeter": row[3]},
            )
        self.stdout.write(self.style.SUCCESS("Successfully uploaded XLSX data."))
