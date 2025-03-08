from django.db import models
from django.contrib.auth.models import User

class Disease(models.Model):
    name = models.CharField(max_length=80, unique=True, blank=False)
    description = models.TextField(blank=True, null=True)
    area = models.TextField(blank=True, null=True) 
    amount = models.TextField(blank=True, null=True) 
    protect=models.TextField(blank=True, null=True)
    severity=models.TextField(blank=True, null=True)

    def __str__(self):
        return self.name
    

class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='predictions/', null=True, blank=True)
    prediction = models.CharField(max_length=255)
    date = models.DateTimeField(auto_now_add=True)

    def str(self):
        return f"{self.user.username} - {self.prediction} - {self.date}"
    
class NearbyShop(models.Model):
    name = models.TextField()
    address = models.TextField()
    phone = models.TextField()
    latitude = models.FloatField()
    longitude = models.FloatField()
    open_time = models.TimeField()
    close_time = models.TimeField()

    def __str__(self):
        return self.name
    
class SeasonalHarvesting(models.Model):
    title = models.CharField(max_length=100, null=True)
    icon = models.CharField(max_length=100, blank=True, null=True)
    season = models.JSONField(default=list, blank=True)
    techniques = models.JSONField(default=list, blank=True)

    def __str__(self):
        return self.title
    
class CropRotation(models.Model):
    name= models.CharField(max_length=100, null=True)
    suitable_months =models.JSONField()
    season_text = models.TextField()
    
    def __str__(self):
      return self.name

class Crop(models.Model):
    name = models.CharField(max_length=100, unique=True)
    growth_duration = models.IntegerField()
    water_per_day = models.FloatField(default=1.0) 

    def __str__(self):
        return self.name