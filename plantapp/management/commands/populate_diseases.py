import pandas as pd
from django.core.management import BaseCommand
from plantapp.models import Disease
from pathlib import Path

class Command(BaseCommand):
    help = "Populate diseases model"

    BASE_PATH = Path(__file__).resolve().parent.parent.parent.parent  # Adjust to your base project path

    def add_arguments(self, parser):
        parser.add_argument('file_name', type=str, help="Name of the excel file")

    def handle(self, *args, **kwargs):
        """Handle what this command does. Every handler to be called here"""
        file_name = kwargs['file_name']
        self.populate_database(file_name)

    def populate_database(self, file_name):
        # Check if the Excel file exists in the project directory
        file_path = self.BASE_PATH / file_name
        if not file_path.exists():
            self.stdout.write(self.style.ERROR(f"Excel file {file_name} doesn't exist at: {self.BASE_PATH}"))
            return None

        self.stdout.write(self.style.WARNING(f"Populating from {file_name}"))

        # Load the Excel file
        df = pd.read_excel(file_path, engine='openpyxl')

        # Strip any leading or trailing whitespaces from column names
        df.columns = df.columns.str.strip()

        # Iterate over rows and populate the Disease model
        for index, row in df.iterrows():
            disease_name = row['Disease Name']  # Use the correct column name
            description = row['Description']  # Ensure the column 'Description' exists
            area = row['Area']
            amount = self.handle_amount(row['Amount'])  # Handle amount with mixed data
            protect = row['How to Naturally/Organically Protect Such Crops']
            severity = row['Disease Severity Assessment']
            
            try:
                # Create a new Disease object
                Disease.objects.create(name=disease_name, description=description, area=area, amount=amount, protect=protect, severity=severity)
                self.stdout.write(self.style.SUCCESS(f"Populated: {disease_name}"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error populating {disease_name}: {e}"))

    def handle_amount(self, amount_value):
        """Handle mixed data in 'amount' column."""
        try:
            # Try to convert to a float if possible
            return float(amount_value) if isinstance(amount_value, (int, float)) or amount_value.replace('.', '', 1).isdigit() else str(amount_value)
        except ValueError:
            # If conversion fails, return as a string
            return str(amount_value)
