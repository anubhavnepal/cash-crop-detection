import os
import pandas as pd
from datetime import datetime
from django.core.management.base import BaseCommand
from plantapp.models import NearbyShop  # Adjust the app name if needed

class Command(BaseCommand):
    help = "Populate the database with nearby shop data from an Excel file."

    def add_arguments(self, parser):
        parser.add_argument('excel_file', type=str, help='Path to the Excel file containing shop data.')

    def handle(self, *args, **options):
        excel_file_path = options['excel_file']

        if not os.path.exists(excel_file_path):
            self.stdout.write(self.style.ERROR(f"File not found: {excel_file_path}"))
            return

        try:
            df = pd.read_excel(excel_file_path)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error reading Excel file: {e}"))
            return

        count = 0
        for _, row in df.iterrows():
            try:
                # Use default values for open/close times if missing
                open_time_str = row.get('Open Time', '09:00')
                close_time_str = row.get('Close Time', '18:00')

                # In case the value is NaN or empty, fallback to defaults
                if pd.isna(open_time_str) or not str(open_time_str).strip():
                    open_time_str = '09:00'
                if pd.isna(close_time_str) or not str(close_time_str).strip():
                    close_time_str = '18:00'

                open_time = datetime.strptime(str(open_time_str), '%H:%M').time()
                close_time = datetime.strptime(str(close_time_str), '%H:%M').time()

                # Adjust the keys based on your Excel headers
                name = row.get('Shop Name')
                address = row.get('Address')
                phone = row.get('Phone Number')

                # If latitude or longitude are missing, set default values (or skip row)
                lat = row.get('Latitude')
                lon = row.get('Longitude')
                if pd.isna(lat) or pd.isna(lon):
                    # For demonstration, we default to 0.0. You may want to skip or handle differently.
                    lat = 0.0
                    lon = 0.0
                else:
                    lat = float(lat)
                    lon = float(lon)

                shop, created = NearbyShop.objects.get_or_create(
                    name=name,
                    defaults={
                        'address': address,
                        'phone': phone,
                        'latitude': lat,
                        'longitude': lon,
                        'open_time': open_time,
                        'close_time': close_time,
                    }
                )
                if created:
                    count += 1
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing row: {row.to_dict()} -- {e}"))
        self.stdout.write(self.style.SUCCESS(f"Successfully populated {count} shop(s)."))
