from django.core.management.base import BaseCommand
from plantapp.models import CropRotation

class Command(BaseCommand):
    help = 'Populate the CropRotation model with initial data'

    def handle(self, *args, **kwargs):
        crops = [
            {
                "name": "potato",
                "suitable_months": [10, 11, 12, 1, 2],
                "season_text": "October to February (Winter crop)"
            },
            {
                "name": "tea",
                "suitable_months": [6, 7, 8, 9],
                "season_text": "June to September (Monsoon season)"
            },
            {
                "name": "coffee",
                "suitable_months": [3, 4, 5, 6, 7, 8],
                "season_text": "March to August (Spring/Summer)"
            },
            {
                "name": "sugarcane",
                "suitable_months": [1,2,3,4,5,6,7,8,9,10,11,12],
                "season_text": "Year-round (Perennial crop)"
            },
        ]

        for crop in crops:
            CropRotation.objects.create(
                name=crop["name"],
                suitable_months=crop["suitable_months"],
                season_text=crop["season_text"]
            )

        self.stdout.write(self.style.SUCCESS('Successfully populated the CropRotation model'))